fex = fex or {}
local function stdnormalpdf(pdf)
	pdf:cmul(pdf)
	pdf:div(-2)
	pdf:exp()
	pdf:mul(1/math.sqrt(2*math.pi))
	return pdf
end

function fex.normpdf(x,mean,std)
	mean = mean or 0
	std = std or 1
	local pdf = x:clone()
	pdf:add(-mean)
	pdf:div(std)
	pdf = stdnormalpdf(pdf)
	pdf:div(std)
	return pdf
end
