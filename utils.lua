
-- convolutions with 'same' option
function fex.conv2(...)
    local arg = {...}
    if arg[#arg] == 'S' then
        local ro,x,k = nil,nil,nil
        if #arg == 4 then
            ro = arg[1]
            x = arg[2]
            k = arg[3]
        else
            x = arg[1]
            k = arg[2]
            ro = x.new(x:size())
        end
        local r = torch.conv2(x,k,'F')
        local shifti = 1+math.ceil((r:size(1)-x:size(1))/2)
        local shiftj = 1+math.ceil((r:size(2)-x:size(2))/2)
        local ii = r:dim()-1
        local jj = r:dim()
        ro:resizeAs(x)
        ro:copy(r:narrow(ii,shifti,x:size(1)):narrow(jj,shiftj,x:size(2)))
        return ro
    else
        return torch.conv2(...)
    end
end

-- cross correlations with 'same' option
function fex.xcorr2(...)
    local arg = {...}
    if arg[#arg] == 'S' then
        local ro,x,k = nil,nil,nil
        if #arg == 4 then
            ro = arg[1]
            x = arg[2]
            k = arg[3]
        else
            x = arg[1]
            k = arg[2]
            ro = x.new(x:size())
        end
        local r = torch.xcorr2(x,k,'F')
        local shifti = 1+math.ceil((r:size(1)-x:size(1))/2)
        local shiftj = 1+math.ceil((r:size(2)-x:size(2))/2)
        local ii = r:dim()-1
        local jj = r:dim()
        ro:resizeAs(x)
        ro:copy(r:narrow(ii,shifti,x:size(1)):narrow(jj,shiftj,x:size(2)))
        return ro
    else
        return torch.xcorr2(...)
    end
end

-- numerical gradient of a tensor.
-- dim is a number that specifies the tensor dimension to calculate gradient
-- dim is a tensor of dimension indices
function fex.gradient(x,dim)

    if not dim then dim = torch.range(0,x:dim()):narrow(1,2,x:dim()) end
    if type(dim) == 'number' then dim = torch.Tensor({dim}) end
    local ndim = x:dim()

    local function grad(x,dim)
        local sz = x:size()
        if sz[dim] == 1 then return x:clone():zero() end
        sz[dim] = sz[dim]+2
        local xx = x.new(sz):zero()
        -- copy center
        xx:narrow(dim,2,x:size(dim)):copy(x)
        -- extrapolate the beginning
        local ff = xx:narrow(dim,1,1)
        local f1 = xx:narrow(dim,2,1)
        local f2 = xx:narrow(dim,3,1)
        torch.add(ff,f1,-1,f2)
        ff:add(f1)
        -- extrapolate the ending
        local xend = xx:size(dim)
        local fe = xx:narrow(dim,xend,1)
        local ff1 = xx:narrow(dim,xend-1,1)
        local ff2 =  xx:narrow(dim,xend-2,1)
        torch.add(fe,ff1,-1,ff2)
        fe:add(ff1)
        -- now subtract
        local d = xx:narrow(dim,3,xend-2):clone()
        d:add(-1,xx:narrow(dim,1,xend-2))
        return d:div(2)
    end

    local res = {}
    for i=1,ndim do
        table.insert(res,i,grad(x,ndim-i+1))
    end
    return unpack(res)
end

local function dimnarrow(x,sz,pad,dim)
    local xn = x
    for i=1,x:dim() do
        if i > dim then
            xn = xn:narrow(i,pad[i]+1,sz[i])
        end
    end
    return xn
end
local function padzero(x,pad)
    local sz = x:size()
    for i=1,x:dim() do sz[i] = sz[i]+pad[i]*2 end
    local xx = x.new(sz):zero()
    local xn = dimnarrow(xx,x:size(),pad,-1)
    xn:copy(x)
    return xx
end
local function padmirror(x,pad)
    local xx = padzero(x,pad)
    local sz  = xx:size()
    for i=1,x:dim() do
        local xxn = dimnarrow(xx,x:size(),pad,i)
        for j=1,pad[i] do
            xxn:select(i,j):copy(xxn:select(i,pad[i]*2-j+1))
            xxn:select(i,sz[i]-j+1):copy(xxn:select(i,sz[i]-pad[i]*2+j))
        end
    end
    return xx
end
function fex.padarray(x,pad,padtype)
    if x:dim() ~= #pad then
        error('number of dimensions of Input should match number of padding sizes')
    end
    if padtype == 'zero' then return padzero(x,pad) end
    if padtype == 'mirror' then return padmirror(x,pad) end
    error('unknown paddtype ' .. padtype)
end




