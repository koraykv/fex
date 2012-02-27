fex = fex or {}

-- This function puts a gaussian pdf with 0 mean unit std deviation
-- on the given input tensor. On entry, the entries should contain
-- the x-coordinates. On exit, it contains corresponding normal pdf 
-- function values.
local function stdnormalpdf(pdf)
	pdf:cmul(pdf)
	pdf:div(-2)
	pdf:exp()
	pdf:mul(1/math.sqrt(2*math.pi))
	return pdf
end

-- This function puts a gaussian pdf with given mean and std deviation
-- on the given input tensor. On entry, the entries should contain
-- the x-coordinates. On exit, it contains corresponding normal pdf 
-- function values.
function fex.normpdf(x,mean,std)
	mean = mean or 0
	std = std or 1
	local pdf = x:clone()
	pdf:add(-mean)
	pdf:div(std)
	stdnormalpdf(pdf)
	pdf:div(std)
	return pdf
end

function fex.gendgauss(sigma)

    --Laplacian of size sigma
    local f_wid = 2 * math.floor(sigma) ;
    local G = fex.normpdf(torch.range(-f_wid,f_wid),0,sigma);
    G = torch.ger(G,G)
    GX,GY = fex.gradient(G);

    GX:div(torch.sum(torch.abs(GX))):mul(2)
    GY:div(torch.sum(torch.abs(GY))):mul(2)
    return GX, GY
end
