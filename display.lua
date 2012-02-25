
function fex.filtersToImage(x,nrow)

    local nfil = 0
    if type(x) == 'userdata' and x:dim() == 2 then
        x={x}
        nfil = 1
    end
    if type(x) == 'table' then
        nfil = #x
    else
        nfil = x:size(1)
    end
    local w,h = x[1]:size(x[1]:dim()),x[1]:size(x[1]:dim()-1)

    local xmax = -math.huge
    for i=1,nfil do
        xmax = math.max(xmax,x[i]:max())
    end
    local nch = x[1]:size(1)
    if x[1]:dim() == 2 then nch = 1 end

    nrow = nrow or 8
    nrow = math.min(nrow,nfil)
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
    -- [0-1] for display
    xx:add(-xx:min())
    xx:div(xx:max())
    return xx
end

function fex.display(xx,ww)
    require 'qtwidget'
    require 'qttorch'
    local xmin = xx:min()
    local w,h = xx:size(xx:dim()),xx:size(xx:dim()-1)
    local ww = ww or qtwidget.newwindow(w,h,'Image Display')
    local qim = qt.QImage.fromTensor(xx)
    ww:image(0,0,qim)
    return ww
end
