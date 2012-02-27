-- SIFT feature extractor as implemented in Svetlana Lazebnik's
-- Pyramid code.
function fex.dsift(im, params)

    local t = torch.tic()

    params = params or {}
    local patch_size = params.patch_size or 16
    local grid_spacing = params.grid_spacing or 1

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
        I = torch.sum(im,1)/3
        I = I[1]
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
    print('1',torch.toc(t))

    local I_theta = torch.atan2(I_Y,I_X)
    I_theta[torch.ne(I_theta,I_theta)]=0
    I_X:cmul(I_X)
    I_Y:cmul(I_Y)
    I_X:add(I_Y)
    local I_mag = I_X:sqrt()
    --local I_mag = torch.sqrt(torch.pow(I_X,2) + torch.pow(I_Y,2))

    local grid_x = torch.range(patch_size/2,wid-patch_size/2+1,grid_spacing)
    local grid_y = torch.range(patch_size/2,hgt-patch_size/2+1,grid_spacing)

    print('2',torch.toc(t))
    local I_orientation = torch.Tensor(num_angles, hgt, wid)

    local cosI = torch.cos(I_theta)
    local sinI = torch.sin(I_theta)

    print('3',torch.toc(t))

    for a=1,num_angles do
        local tmp = cosI*math.cos(angles[a]) + sinI*math.sin(angles[a])
        tmp:pow(alpha)
        tmp[torch.le(tmp,0)] = 0
        torch.cmul(I_orientation[a], tmp, I_mag)
    end
    print('4',torch.toc(t))

    local weight_kernel = torch.zeros(patch_size,patch_size)
    local r = patch_size/2;
    local cx = r-0.5;
    local sample_res = patch_size/num_bins;
    local weight_x = torch.abs(torch.range(1,patch_size)-cx)/sample_res
    weight_x:apply(function(x) if x <= 1 then return 1-x else return 0 end end)
    local tw = torch.ger(weight_x, weight_x)
    print('5',torch.toc(t))
    local wx= torch.Tensor(weight_x):resize(weight_x:size(1),1)

    --local I_orientation2 = torch.Tensor(num_angles, hgt-wx:size(1)+1, wid-wx:size(1)+1)
    local I_orientation2 = torch.Tensor():resizeAs(I_orientation)
    for a=1,num_angles do
        local t = fex.conv2(I_orientation[a],wx,'S')
        fex.conv2(I_orientation2[a],t,wx:t(),'S')
        --fex.conv2(I_orientation2[a],I_orientation[a],tw,'S')
    end
    print('6',torch.toc(t))
    return I_orientation2
end


