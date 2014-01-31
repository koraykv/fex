

local function csize(x,k,s)
    local o = math.floor((x-k)/s+1)
    local xi = (o-1)*s+k
    return xi,o
end
-- patch_size is the region over which the SIFT feature is computed
-- grid_spacing is the step size between each patch
-- num_bins is the 
local function sift_sampler(io,patch_size,grid_spacing,num_bins)
    local num_angles = io:size(1)
    local binstep = math.floor(patch_size/num_bins)
    local nr,nor = csize(io:size(2),patch_size,grid_spacing)
    local nc,noc = csize(io:size(3),patch_size,grid_spacing)
    local shiftr = 1+math.floor((io:size(2)-nr)/2)
    local shiftc = 1+math.floor((io:size(3)-nc)/2)
    local nio = io:narrow(2,shiftr,nr):narrow(3,shiftc,nc)
    local uio = nio:unfold(2,patch_size,grid_spacing):unfold(3,patch_size,grid_spacing)
    local sift = torch.Tensor(num_bins*num_bins*num_angles,nor,noc)
    local b = 1
    for i=1,num_bins do
        for j=1,num_bins do
            sift:narrow(1,b,num_angles):copy(uio:select(5,(i-1)*binstep+1):select(4,(j-1)*binstep+1))
            b = b + num_angles
        end
    end
    return sift
end

local function sift_normalize(sift)
    error('not yet implemented')
end
local function sift_normalize_dense(sift)
    local ct = .1;
    torch.add(sift,sift,ct)
    local tmp = torch.cmul(sift,sift)
    local tmp2 = torch.sum(tmp,1)
    torch.sqrt(tmp2,tmp2)
    -- print(tmp2[{ 1 , {1,10} , {1,10} }])
    local rtmp2 = torch.Tensor(tmp2:storage(),tmp2:storageOffset(),
                               sift:size(1),0,tmp2:size(2),tmp2:stride(2),tmp2:size(3),tmp2:stride(3))

    -- print(rtmp2:size())
    torch.cdiv(sift,sift,rtmp2)
    return sift
end


-- SIFT feature extractor as implemented in Svetlana Lazebnik's
-- Pyramid code.
function fex.dsift(im, params)

    local t = torch.tic()

    params = params or {}
    local patch_size = params.patch_size or 16
    local grid_spacing = params.grid_spacing or 8

    local num_angles = 8
    local num_bins = 4
    local num_samples = num_bins * num_bins
    local alpha = 9
    local sigma_edge = 1

    local angle_step = 2 * math.pi / num_angles
    local angles = torch.range(0,2*math.pi,angle_step)
    angles = angles:narrow(1,1,num_angles)

    local I
    if im:dim() == 3 then
        I = torch.Tensor(im:size(2),im:size(3))
        torch.add(I,im[1],im[2])
        I:add(im[3])
        I:div(3)
    elseif im:dim() == 2 then
        I = im
    else
        error('im has to be 2D or 3D')
    end
    I:div(I:max())
    local hgt = I:size(1)
    local wid = I:size(2)

    local G_X,G_Y = fex.gendgauss(sigma_edge)
    
    I = fex.padarray(I,{2,2},'mirror')
    I:add(-I:mean())

    local I_X = fex.xcorr2(I, G_X, 'S')
    local I_Y = fex.xcorr2(I, G_Y, 'S')

    I_X = I_X:narrow(1,3,hgt):narrow(2,3,wid):clone()
    I_Y = I_Y:narrow(1,3,hgt):narrow(2,3,wid):clone()
    if fex.verbose then print('1',torch.toc(t)) end

    local I_theta = torch.atan2(I_Y,I_X)
    I_theta[torch.ne(I_theta,I_theta)]=0
    I_X:cmul(I_X)
    I_Y:cmul(I_Y)
    I_X:add(I_Y)
    local I_mag = I_X:sqrt()
    --local I_mag = torch.sqrt(torch.pow(I_X,2) + torch.pow(I_Y,2))

    local grid_x = torch.range(patch_size/2,wid-patch_size/2+1,grid_spacing)
    local grid_y = torch.range(patch_size/2,hgt-patch_size/2+1,grid_spacing)

    if fex.verbose then print('2',torch.toc(t)) end
    local I_orientation = torch.Tensor(num_angles, hgt, wid)

    local cosI = torch.cos(I_theta)
    local sinI = torch.sin(I_theta)

    if fex.verbose then print('3',torch.toc(t)) end

    local tmp = torch.Tensor(num_angles,hgt,wid)
    for a=1,num_angles do
        torch.mul(tmp[a],cosI,math.cos(angles[a]))
        tmp[a]:add(math.sin(angles[a]),sinI)
    end
    tmp:pow(alpha)
    tmp[torch.le(tmp,0)] = 0
    local tt = torch.Tensor(I_mag:storage(),I_mag:storageOffset(),num_angles,0,hgt,I_mag:stride(1),wid,I_mag:stride(2))
    torch.cmul(I_orientation,tmp,tt)

    if fex.verbose then print('4',torch.toc(t)) end

    local weight_kernel = torch.zeros(patch_size,patch_size)
    local r = patch_size/2;
    local cx = r-0.5;
    local sample_res = patch_size/num_bins;
    local weight_x = torch.abs(torch.range(1,patch_size)-cx)/sample_res
    weight_x:apply(function(x) if x <= 1 then return 1-x else return 0 end end)

    if fex.verbose then print('5',torch.toc(t)) end

    local wx= torch.Tensor(weight_x):resize(weight_x:size(1),1)

    local I_orientation2 = torch.Tensor():resizeAs(I_orientation)
    for a=1,num_angles do
        local t = fex.conv2(I_orientation[a],wx:t(),'S')
        fex.conv2(I_orientation2[a],t,wx,'S')
    end

    if fex.verbose then print('6',torch.toc(t)) end

    local sift=sift_sampler(I_orientation2,patch_size,grid_spacing,num_bins)
    if fex.verbose then print('7',torch.toc(t)) end
    sift = sift_normalize_dense(sift)
    collectgarbage()
    if fex.verbose then print('8',torch.toc(t)) end
    return sift
end

-- SIFT feature extractor as implemented in Svetlana Lazebnik's
-- Pyramid code.
function fex.dsiftfast(im, params)

    local t = torch.tic()

    params = params or {}
    local patch_size = params.patch_size or 16
    local grid_spacing = params.grid_spacing or 8

    local num_angles = 8
    local num_bins = 4
    local num_samples = num_bins * num_bins
    local alpha = 9
    local sigma_edge = 1

    local angle_step = 2 * math.pi / num_angles
    local angles = torch.range(0,2*math.pi,angle_step)
    angles = angles:narrow(1,1,num_angles)

    local I
    if im:dim() == 3 then
        I = torch.Tensor(im:size(2),im:size(3))
        torch.add(I,im[1],im[2])
        I:add(im[3])
        I:div(3)
    elseif im:dim() == 2 then
        I = im
    else
        error('im has to be 2D or 3D')
    end

    I:div(I:max())

    local hgt,wid = I:size(1),I:size(2)
    local G_X,G_Y = fex.gendgauss(sigma_edge)
    
    I:add(-I:mean())
    if fex.verbose then print('0',torch.toc(t)) end
    local I_X = torch.xcorr2(I, G_X)
    local I_Y = torch.xcorr2(I, G_Y)

    if fex.verbose then print('1',torch.toc(t)) end

    local I_theta = torch.atan2(I_Y,I_X)
    I_theta[torch.ne(I_theta,I_theta)]=0
    I_X:cmul(I_X)
    I_Y:cmul(I_Y)
    I_X:add(I_Y)
    local I_mag = I_X:sqrt()
    hgt,wid = I_mag:size(1),I_mag:size(2)

    if fex.verbose then print('2',torch.toc(t)) end
    local I_orientation = torch.Tensor(num_angles, hgt, wid)
    local cosI = torch.cos(I_theta)
    local sinI = torch.sin(I_theta)

    if fex.verbose then print('3',torch.toc(t)) end

    local tmp = torch.Tensor(num_angles,hgt,wid)
    for a=1,num_angles do
        torch.mul(tmp[a],cosI,math.cos(angles[a]))
        tmp[a]:add(math.sin(angles[a]),sinI)
    end
    tmp:pow(alpha)
    tmp[torch.le(tmp,0)] = 0
    local tt = torch.Tensor(I_mag:storage(),I_mag:storageOffset(),num_angles,0,hgt,I_mag:stride(1),wid,I_mag:stride(2))
    torch.cmul(I_orientation,tmp,tt)
    if fex.verbose then print('4',torch.toc(t)) end

    local weight_kernel = torch.zeros(patch_size,patch_size)
    local r = patch_size/2;
    local cx = r-0.5;
    local sample_res = patch_size/num_bins;
    local weight_x = torch.abs(torch.range(1,patch_size)-cx)/sample_res
    weight_x:apply(function(x) if x <= 1 then return 1-x else return 0 end end)
    weight_x = torch.Tensor(weight_x):resize(weight_x:size(1),1)

    if fex.verbose then print('5',torch.toc(t)) end

    local I_orientation2 = torch.Tensor(num_angles, hgt-weight_x:size(1)+1, wid-weight_x:size(1)+1)
    --local I_orientation2 = torch.Tensor():resizeAs(I_orientation)
    local tim = torch.Tensor(num_angles, I_orientation2:size(2), wid)
    for a=1,num_angles do
        torch.conv2(tim[a],I_orientation[a],weight_x)
        torch.conv2(I_orientation2[a],tim[a],weight_x:t())
    end
    if fex.verbose then print('6',torch.toc(t)) end
    local sift=sift_sampler(I_orientation2,patch_size,grid_spacing,num_bins)
    if fex.verbose then print('7',torch.toc(t)) end
    sift = sift_normalize_dense(sift)
    collectgarbage()
    if fex.verbose then print('8',torch.toc(t)) end
    return sift
end


