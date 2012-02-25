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
