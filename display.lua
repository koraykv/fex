
function fex.imToDisplay(x,params)

    params = params or {}
    if params.sym == nil then params.sym = false end
    params.nrow = params.nrow or 8

    if type(x) == 'table' then
        local xx = x[1]
        local sz = torch.LongStorage(4)
        sz[1] = #x
        if xx:dim() == 2 then sz[2] = 1 else sz[2] = xx:size(1) end
        sz[3] = xx:size(xx:dim()-1)
        sz[4] = xx:size(xx:dim())
        local xx = torch.Tensor(sz)
        for i=1,#x do xx[i]:copy(x[i]) end
        x=xx
    end
    if x:dim() == 2 then
        x=torch.Tensor(x):resize(1,x:size(1),x:size(2))
    end
    if x:dim() == 3 then
        if x:size(1) == 3 then
            x=torch.Tensor(x):resize(1,x:size(1),x:size(2),x:size(3))
        else
            x=torch.Tensor(x):resize(x:size(1),1,x:size(2),x:size(3))
        end
    end
    if x:dim() ~= 4 then error('WTF') end
    local nfil,nch,h,w = x:size(1),x:size(2),x:size(3),x:size(4)

    local xmax = -math.huge
    if params.sym then
        xmax = math.max(math.abs(x:max()),math.abs(x:min()))
    else
        xmax = x:max()
    end

    local nrow = math.min(params.nrow,nfil)
    local ncol = math.floor(nfil/nrow) + math.min(nfil % nrow, 1)
    local xx = torch.Tensor(nch, ncol * (h+1) + 1 , nrow * (w+1) + 1)
    xx:fill(xmax)
    local ii = 1
    for j=1,ncol do
        local xloc = (h+1)*(j-1)+2
        for i=1,nrow do
            local yloc = (w+1)*(i-1)+2
            xx:narrow(2,xloc,h):narrow(3,yloc,w):copy(x[ii])
            ii = ii + 1
            if ii>nfil then break end
        end
    end
    if params.sym then
        xx:add(xmax)
        xx:div(2*xmax)
    else
        -- [0-1] for display
        xx:add(-xx:min())
        xx:div(xx:max())
    end
    return xx
end

function fex.imshow(im,params)
    require 'qtwidget'
    require 'qttorch'
    params = params or {}
    local xx = fex.imToDisplay(im,params)
    local w,h = xx:size(xx:dim()),xx:size(xx:dim()-1)
    local ww = params.ww or qtwidget.newwindow(w,h,'Image Display')
    local xi = params.x or 0
    local yi = params.y or 0
    local qim = qt.QImage.fromTensor(xx)
    local wr,hr = ww:currentsize()
    local ss = math.min(wr/w,hr/h)
    local wi,hi = w*ss,h*ss
    ww:resize(wi,hi)
    ww:image(xi,yi,wi,hi,qim)
    ww:onResize(function(w,h) ww:image(xi,yi,w,h,qim) end)
    return ww
end


