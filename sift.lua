
function gendgauss(sigma)

    --Laplacian of size sigma
    local f_wid = 4 * math.floor(sigma);
    local G = fex.normpdf(torch.range(-f_wid,f_wid),0,sigma);
    G = torch.ger(G,G)
    GX,GY = fex.gradient(G);

    GX:div(torch.sum(torch.abs(GX))):mul(2)
    GY:div(torch.sum(torch.abs(GY))):mul(2)
    return GX, GY
end

function fex.dsift(im, grid_spacing, patch_size)
    local I
    if im:dim() == 3 then
        I = torch.sum(im,1)/3
        I = I[1]
    elseif im:dim() == 2 then
        I = im
    else
        error('im has to be 2D or 3D')
    end

    patch_size = patch_size or 16
    grid_spacing = grid_spacing or patch_size/2

    local num_angles = 8
    local num_bins = 4
    local num_samples = num_bins * num_bins

    local alpha = 9
    local sigma_edge = 1

    local angle_step = 2 * math.pi / num_angles
    local angles = torch.range(0,2*math.pi,angle_step)
    angles = angles:narrow(1,1,num_angles):clone()

    local hgt = I:size(1)
    local wid = I:size(2)

    local G_X,G_Y = gendgauss(sigma_edge)
    
    I:add(-torch.mean(I))
    local I_X = fex.xcorr2(I, G_X, 'S')
    local I_Y = fex.xcorr2(I, G_Y, 'S')

    local I_mag = torch.sqrt(torch.pow(I_X,2) + torch.pow(I_Y,2))
    local I_theta = torch.atan(torch.cdiv(I_Y,I_X))
    I_theta[torch.ne(I_theta,I_theta)]=0

    local grid_x = torch.range(patch_size/2,wid-patch_size/2,grid_spacing)
    local grid_y = torch.range(patch_size/2,hgt-patch_size/2,grid_spacing)

    local I_orientation = torch.zeros(num_angles, hgt, wid)

    local cosI = torch.cos(I_theta)
    local sinI = torch.sin(I_theta)
    
    for a=1,num_angles do
        local tmp = cosI*math.cos(angles[a]) + sinI*math.sin(angles[a])
        tmp:pow(alpha)
        tmp[torch.le(tmp,0)] = 0
        torch.cmul(I_orientation[a], tmp, I_mag)
    end

    local weight_kernel = torch.zeros(patch_size,patch_size)
    local r = patch_size/2;
    local cx = r-0.5;
    local sample_res = patch_size/num_bins;
    local weight_x = torch.abs(torch.range(1,patch_size)-cx)/sample_res
    weight_x:apply(function(x) if x <= 1 then return 1-x else return x end end)
    local tw = torch.ger(weight_x, weight_x)

    for a=1,num_angles do
        I_orientation[a]:copy(fex.conv2(I_orientation[a],tw,'S'))
    end
    return I_orientation
end














