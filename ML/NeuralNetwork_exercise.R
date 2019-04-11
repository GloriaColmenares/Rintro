reset_par <- function(){
  op <- structure(list(xlog = FALSE, ylog = FALSE, adj = 0.5, ann = TRUE,
                       ask = FALSE, bg = "transparent", bty = "o", cex = 1, cex.axis = 1,
                       cex.lab = 1, cex.main = 1.2, cex.sub = 1, col = "black",
                       col.axis = "black", col.lab = "black", col.main = "black",
                       col.sub = "black", crt = 0, err = 0L, family = "", fg = "black",
                       fig = c(0, 1, 0, 1), fin = c(6.99999895833333, 6.99999895833333
                       ), font = 1L, font.axis = 1L, font.lab = 1L, font.main = 2L,
                       font.sub = 1L, lab = c(5L, 5L, 7L), las = 0L, lend = "round",
                       lheight = 1, ljoin = "round", lmitre = 10, lty = "solid",
                       lwd = 1, mai = c(1.02, 0.82, 0.82, 0.42), mar = c(5.1, 4.1,
                                                                         4.1, 2.1), mex = 1, mfcol = c(1L, 1L), mfg = c(1L, 1L, 1L,
                                                                                                                        1L), mfrow = c(1L, 1L), mgp = c(3, 1, 0), mkh = 0.001, new = FALSE,
                       oma = c(0, 0, 0, 0), omd = c(0, 1, 0, 1), omi = c(0, 0, 0,
                                                                         0), pch = 1L, pin = c(5.75999895833333, 5.15999895833333),
                       plt = c(0.117142874574832, 0.939999991071427, 0.145714307397962,
                               0.882857125425167), ps = 12L, pty = "m", smo = 1, srt = 0,
                       tck = NA_real_, tcl = -0.5, usr = c(0.568, 1.432, 0.568,
                                                           1.432), xaxp = c(0.6, 1.4, 4), xaxs = "r", xaxt = "s", xpd = FALSE,
                       yaxp = c(0.6, 1.4, 4), yaxs = "r", yaxt = "s", ylbias = 0.2), .Names = c("xlog",
                                                                                                "ylog", "adj", "ann", "ask", "bg", "bty", "cex", "cex.axis",
                                                                                                "cex.lab", "cex.main", "cex.sub", "col", "col.axis", "col.lab",
                                                                                                "col.main", "col.sub", "crt", "err", "family", "fg", "fig", "fin",
                                                                                                "font", "font.axis", "font.lab", "font.main", "font.sub", "lab",
                                                                                                "las", "lend", "lheight", "ljoin", "lmitre", "lty", "lwd", "mai",
                                                                                                "mar", "mex", "mfcol", "mfg", "mfrow", "mgp", "mkh", "new", "oma",
                                                                                                "omd", "omi", "pch", "pin", "plt", "ps", "pty", "smo", "srt",
                                                                                                "tck", "tcl", "usr", "xaxp", "xaxs", "xaxt", "xpd", "yaxp", "yaxs",
                                                                                                "yaxt", "ylbias"))
  par(op)
}




####Neuronal Network to recognize handwritten digits####



## Creats a neural network with length(sizes) layers. The first layer is an input layer and only distributes the inputs to the second layer.
## Hence, neither weights nor biases are computed for the first layer.
## The gradient entries of the list (elements [[4]] and [[5]]) are empty dummies that only represent the structure of the gradient.
net.create <- function(sizes){
  net <- list(sizes = sizes)
  net[[2]] <- list()
  net[[2]][[1]] <- matrix(rnorm(sizes[1]), ncol = 1)
  net[[3]] <- list()
  net[[4]] <- list()
  net[[5]] <- list()
  for(i in 1:length(sizes)){
    if(i == 1){
      net[[2]][[i]] <- matrix(rep(0, sizes[i]), nrow = sizes[i])
    }else{
      net[[2]][[i]] <- matrix(rnorm(sizes[i]*sizes[i-1], sd = 1/sqrt(sizes[i-1])), nrow = sizes[i])
    }
    net[[3]][[i]] <- matrix(rnorm(sizes[i]), nrow = sizes[i])*(i!=1)
    net[[4]][[i]] <- matrix(rnorm(sizes[i]*sizes[i-1]), nrow = sizes[i])*0
    if (i == 1){net[[4]][[i]] <- matrix(rnorm(sizes[i]), nrow = sizes[i])*0}
    net[[5]][[i]] <- matrix(0, nrow = sizes[i])*(i!=1)*0
  }
  names(net) <- c("sizes", "weights", "biases", "grad_w", "grad_b")
  #Gradient computation missing!
  return(net)
}

#### Computes the output from a neural network
#### Inputs are a neural network (net) and a single data point (data), i.e. in the case of digits a single image
#### If "full" is set to T, all intermediate output is returned, as well. This is useful to plot the whole network
net.IO <- function(net, data, full = F){
  biases <- rep(0, net$sizes[1])
  output <- rep(0, net$sizes[1])
  input  <- data
  if(!full){
    for(i in 2:length(net$sizes)){
      output <- rep(0, net$sizes[i])
      for(j in 1:net$sizes[i]){
        output[j] <- net.sigmoid(weights = net$weights[[i]][j,], biases = net$biases[[i]][j,], data = input)
      }
      input <- output
    }
    return(output)
  }else{
    outputs <- list(input)
    for(i in 2:length(net$sizes)){
      output <- rep(0, net$sizes[i])
      for(j in 1:net$sizes[i]){
        output[j] <- net.sigmoid(weights = net$weights[[i]][j,], biases = net$biases[[i]][j,], data = input)
      }
      outputs[[i]] <- output
      input <- output
    }
    return(outputs)
    
  }
}


### The sigmoid activation function. 
### Inputs are the weights and biases of the current neuron or layer as well as all inputs (data)
### Weights is a matrix of dimension N_neurons x N_inputs
### Biases is a matrix of dimension N_neurons x 1
### Data is a vector of length(N_inputs)
### or a vector()
net.sigmoid <- function(weights, biases, data){
  return(1/(1+exp(-(weights%*%data+biases))))
}


### Derivative of the sigmoid function wrt. z = t(weights)%*%data + biases
net.sigmoid.deriv <- function(weights, biases, data){
  net.sigmoid.eval <- net.sigmoid(weights, biases, data)
  return(net.sigmoid.eval*(1-net.sigmoid.eval))
}



### The cost function of the neural network. Distinguishe between squares, ie c = ||data-net.output|| and crossentropy
### crossentropy increases learning speed in early epochs.
### Inputs are a data vector, a forecast vector (net.output) and the cost function that's used (cf; default is "squares")
net.cost    <- function(data, net.output, cf = "squares"){
  if(cf == "squares"){
    if(length(data)==1){
      dta         <- rep(0, length(net.output))
      dta[data+1] <- 1
      data        <- dta
    }
    return(sum((data-net.output)^2))
  }
  if(cf == "ce"){ #cross entropy
    if(length(data)==1){
      dta         <- rep(0, length(net.output))
      dta[data+1] <- 1
      data        <- dta
    }
    return(-sum(data*log(net.output)+(1-data)*log(1-net.output)))
  }
}


### Derivative of the cost function wrt. net.output
### Inputs are a data vector, a forecast vector (net.output) and the cost function that's used (cf; default is "squares")
net.cost.prime <- function(data, net.output, cf = "squares"){ #data = true data. Z.B. wenn die Ziffer eine 2 ist: (0,0,1,0,0,0,0,0,0,0)
  if(cf == "squares"){
    if(length(data)==1){
      dta         <- rep(0, length(net.output))
      dta[data+1] <- 1
      data        <- dta
    }
    return(2*(net.output-data))
  }
  if(cf == "ce"){
    if(length(data)==1){
      dta         <- rep(0, length(net.output))
      dta[data+1] <- 1
      data        <- dta
    }
    return(-(data / net.output - (1 - data) / (1 - net.output)))
  }
}

### Compute the fraction of correctly identified digits. 
### Inputs are a (trained) neural networt (net) and a list of testdata; the first entry is a list of images data, the second entry is the respective digit
net.evaluate <- function(net, test.data){
  listoutput <- rep(0, length(test.data[[2]]))
  for(i in 1:nrow(test.data[[1]])){
    listoutput[i] <- which.max(net.IO(net, test.data[[1]][i,]))-1
  }
  return(mean(listoutput==test.data[[2]]))
}



### A wrapper for SGD to be used in lapply()
### Inputs are a data list: a list with each entry being a list with a full training data set consisting of a list with images data and digits.
### Other inputs are: a net that is to be trained, the number of training epochs, the size of the mini bath in SGD, the learning rate eta, the penalty lambda
### a char to identify the cost function, a booolean ISP to compute in-sample-precision (ISP) 
### and an integer ISP.interval to determine the frequency of ISP computations (ISP.interval)
net.SGD.wrapper <- function(data, net, epochs, mini.batch.size, eta, lambda = 0, cf = "squares", ISP = F, ISP.interval = epochs){
  training.data <- data[[1]]
  test.data     <- data[[2]]
  return(net.SGD(net = net, training.data = training.data, epochs = epochs, mini.batch.size = mini.batch.size,
                 eta = eta, lambda = lambda, test.data = test.data, cf = cf, ISP = ISP, ISP.interval = ISP.interval ))
}


### Basic function to train a neural net using stochastic gradient descent (SGD)
### Inputs are a net that is to be trained, a list of training data, containing a matrix of images data and a vector of digits
### the number of training epochs, the size of the mini bath in SGD, the learning rate eta, the penalty lambda
### a list of test data (in the same fashion as training data)
### a char to identify the cost function, a booolean ISP to compute in-sample-precision (ISP) 
### and an integer ISP.interval to determine the frequency of ISP computations (ISP.interval)
### ISP slows down the training, substantially
net.SGD <- function(net, training.data, epochs, mini.batch.size, eta, lambda = 0, test.data = NULL, cf = "squares", ISP = F, ISP.interval = epochs){
  if(!is.null(test.data)){
    eval.max  <- net.evaluate(net = net, test.data = test.data)
    maxnet    <- net
    i.max     <- 1
  }
  N             <- length(training.data[[2]])
  train.img     <- training.data[[1]]
  train.trueval <- training.data[[2]]
  usedlen       <- length(train.trueval)%/%mini.batch.size*mini.batch.size  ##Wie viele komplette Minibatches können ausgeführt werden?
  train.img     <- train.img[1:usedlen,]
  train.trueval <- train.trueval[1:usedlen]
  for(i in 1:epochs){
    shuffled   <- sample(1:usedlen)
    for(j in 1:(usedlen/mini.batch.size)){
      mb.img     <- train.img[shuffled[((j-1)*mini.batch.size+1):(j*mini.batch.size)],]
      mb.trueval <- train.trueval[shuffled[((j-1)*mini.batch.size+1):(j*mini.batch.size)]]
      meangrad   <- net.mini.batch.update(net = net, minibatch = list(mb.img, mb.trueval), cf = cf)
      for(k in 1:length(net$weights)){
        net$weights[[k]] <- (1-(eta*lambda)/N)*net$weights[[k]] - meangrad[[2]][[k]] * eta
      }
      for(k in 1:length(net$biases)){
        net$biases[[k]]  <- net$biases[[k]]  - meangrad[[1]][[k]] * eta
      }
    }
    if(is.null(test.data)){
      print(paste("Epoch:", i, "time:", Sys.time()))
    }else{
      eval <- net.evaluate(net = net, test.data = test.data)
      if(eval > eval.max){
        eval.max <- eval
        maxnet   <- net
      }
      if(!ISP|(i%%ISP.interval!=0)){
        print(paste0("Epoch: ", i, "; time: ", Sys.time(), "; OoS Precision: ", round(eval*100,2),"%"))
      }else{
        eval.I <- net.evaluate(net = net, test.data = training.data)
        print(paste0("Epoch: ", i, "; time: ", Sys.time(), "; OoS Precision: ", round(eval*100,2),"%", "; iSP: ", round(eval.I*100,2),"%" ))
      }
    }
  }
  if(is.null(test.data)){return(net)}else{return(maxnet)}
}


### Compute the gradient of the cost function wrt w and b
### Inputs are the (trained) net, a list of data, containing a matrix of image inputs and a vector of digits and a char to determine the cost function
### ToDo: first layer gradient = 0, is there a better way to code this?

net.gradient <- function(net, data, cf = "squares"){
  net.size         <- length(net$sizes)
  grad_w           <- net$grad_w
  grad_b           <- net$grad_b
  activations      <- list()
  input            <- data[[1]]
  trueval          <- data[[2]]
  activations[[1]] <- input
  activations[[2]] <- input #rep(0,net$sizes[1])
  
  #for(j in 1:net$sizes[1]){
  #  activations[[2]][j] <- net.sigmoid(weights = net$weights[[1]][j,], biases = net$biases[[1]][j,], data = input[j])
  #}
  for(i in 2:net.size){
    activations[[i+1]] <- net.sigmoid(weights = net$weights[[i]], biases = net$biases[[i]], data = activations[[i]])
  }
  if(cf == "ce"){
    activations[[net.size+1]] <- pmax(pmin(activations[[net.size+1]],0.9999999), 0.0000001)
  }
  net.output             <- activations[[net.size+1]]
  delta                  <- net.cost.prime(data = trueval, net.output = activations[[net.size+1]], cf = cf) *activations[[net.size+1]]*(1-activations[[net.size+1]])   #* net.sigmoid.deriv(weights = net$weights[[net.size]], biases = net$biases[[net.size]], data = activations[[net.size]])
  grad_b[[net.size]][,1] <- delta
  grad_w[[net.size]]     <- delta%*%t(activations[[net.size]])
  for(i in (net.size-1):2){
    #delta <- (t(net$weights[[i+1]])%*%delta)*net.sigmoid.deriv(weights = net$weights[[i]], biases= net$biases[[i]], data = activations[[i]] )
    delta       <- (t(net$weights[[i+1]])%*%delta) * (activations[[i+1]] * (1-activations[[i+1]]))
    grad_b[[i]] <- delta
    grad_w[[i]] <- delta%*%t(activations[[i]])
  }
  grad_b[[1]] <- grad_b[[1]]*0
  grad_w[[1]] <- grad_b[[1]]*0
  return(list(grad_b = grad_b,grad_w = grad_w))
}


### Computes the average gradient for a single mini batch in stochastic gradient descent.
### Inputs are a (trained) net, the minibatch, ie. a list containing a matrix with images data and a vector with digits 
### and a char to determine the cost function
net.mini.batch.update <- function(net, minibatch, cf = "squares"){ 
  minibatch.img <- minibatch[[1]]
  minibatch.dig <- minibatch[[2]]
  gradlist      <- list()
  for(i in 1:length(minibatch.dig)){
    gradlist[[i]] <- net.gradient(net = net, data = list(minibatch.img[i,], minibatch.dig[i]), cf = cf)
  }
  meangrad <- gradlist[[1]]
  for(i in 2:length(minibatch.dig)){
    for(j in 1:length(gradlist[[i]])){
      for(k in 1:length(gradlist[[i]][[j]])){
        meangrad[[j]][[k]] <- meangrad[[j]][[k]] + gradlist[[i]][[j]][[k]]
      }
    }
  }
  for(j in 1:length(gradlist[[i]])){
    for(k in 1:length(gradlist[[i]][[j]])){
      meangrad[[j]][[k]] <- meangrad[[j]][[k]]/length(minibatch.dig)
    }
  }
  return(meangrad)
}

### Plots the digits from the data set.
### inputs are a vector of image data and additional arguments for plot()

digitplot <- function(digit, nrows = sqrt(length(digit)), ncols = sqrt(length(digit)), trueval = NA, predictval = NA, ...){
  oldpars      <- par(c("mai", "omi", "mgp"))
  par(mai = c(0,0,0,0), omi = c(0,0,0,0),mgp = c(0,0,0))
  if(max(digit)<=1){digit=digit*255}
  digitplot.inner(digit = digit, nrows = nrows, ncols = ncols, trueval = trueval, predictval = predictval, ...)
  par(oldpars)
} 


digitplot.inner <- function(digit, nrows = sqrt(length(digit)), ncols = sqrt(length(digit)), trueval = NA, predictval = NA, ...){
  if(max(digit)<=1){digit=digit*255}
  if(!(round(nrows)==nrows&round(ncols)==ncols)){stop("Please define square image dimensions")}
  nrows = sqrt(length(digit)) 
  ncols = sqrt(length(digit))
  grid_img     <- matrix(nrow = nrows * ncols, ncol=2)
  grid_img[,1] <- rep(1:nrows, each = ncols)
  grid_img[,2] <- rep(ncols:1, nrows)
  greyscale    <- grey.colors(n =256, start = 1, end = 0)
  plot(grid_img, col=greyscale[digit+1], pch=15, ylab ="", xlab ="", xaxt ="n", yaxt ="n",bty ="n" , ... )
  if(!is.na(trueval)&!is.na(predictval)){
    if(predictval==trueval){
      titlecol ="green"
    }else{titlecol = "red"}
    title(main = paste0("True value: ", trueval, "; Predicted value:", predictval), col.main = titlecol)
  }else{
    if(!is.na(trueval)){
      title(main = paste0("True value: ", trueval))
    }
    if(!is.na(predictval)){
      title(main = paste0("Predicted value: ", predictval))
    }
  }
}



### Plot the neural network
net.plot<- function(net, plot.sizes = NA, spread = 0.75, full = F, full.input = T, col.list = NA, ...){
  oldpars <- par(c("mar", "mgp"))           ## save old par() values to restore, later
  newpars <- oldpars
  newpars$mar <- c(0,0,1,0)
  newpars$oma <- c(0,0,1,0)
  par(newpars)
  net.plot.inner(net = net, plot.sizes = plot.sizes, spread = spread, full = full, full.input = full.input, col.list = col.list, ...)  
  par(oldpars)
}



### This is mainly used from whithin net.IO.plot and only plots a meaningful plot if margins are set manually, before.
net.plot.inner <- function(net, plot.sizes = NA, spread = 0.75, full = F, full.input = T, col.list = NA, ...){
  n.hidden <- length(net$sizes)-2           ## compute the number of hidden layers in the networ. All layers that are not input or output are hidden.
  center.hidden <- (length(net$sizes)+1)/2  ## compute the center (x axis) of the hidden layers to label them, later  ys <- 0
  if(any(is.na(plot.sizes))){               ## if there is no rule, which nodes are to be plotted, plot full input and output layers
    plot.sizes <- net$sizes
    if(!full){ ## if not every node should be plotted, plot at most three nodes.
      plot.sizes[-c(1,length(plot.sizes))]<- pmin(3, plot.sizes[-c(1,length(plot.sizes))]) 
    }
  }
  ys <- 1
  xs <- 1:length(plot.sizes)
  y.range <- c(1,max(plot.sizes))
  y.range.old <- y.range
  y.center  <- max(y.range+1)/2
  #print(y.center)
  y.range <- c((1-0.95)*y.center, (1+0.95)*y.center)  
  y.rescale <- c((1-spread)*y.center, (1+spread)*y.center)     ### Bereich, in dem alle nicht vollständig gezeichneten Schichten des Netzes gezeichnet werden.
  plot(xs, rep(1, length(xs)), type = "n",..., ylim = range(-1, y.range+1), xlim = range(xs, xs-1, xs+1),  bty ="n", axes = F, main =paste0("Neural Network with ", length(net$sizes), " layers"))
  for(i in 1:length(xs)){
    text(i,y.range.old[2], paste(net$sizes[i], "neurons"))
    oldys <- ys
    if(!i%in%c(1,length(plot.sizes))){
      #if(plot.sizes[i]==net$sizes[i]){
      #  ys <- seq(from = y.range[1], to = y.range[2], length = plot.sizes[i]) 
      #}else{
      ys <- seq(from = y.rescale[1], to = y.rescale[2], length.out = plot.sizes[i])
      #}
    }else{
      ys <- seq(from = y.range[1], to = y.range[2], length = plot.sizes[i])
      if(i == 1 & full.input == F){
        ys <- c(seq(from = y.range[1], to = y.range[1]+0.4*mean(y.range), length = 5), seq(from = y.range[2]-0.4*mean(y.range), to = y.range[2], length = 5))
      }
    }
    dist <- diff(ys)[1]
    if(length(ys)==1){ys = mean(y.rescale); dist = 0}
    if(i == 1){
      for(n in 1:length(ys)){
        lines(c(i-0.2,i), c(ys[n], ys[n]), lty = "dashed")
      }
    }
    if(i == length(xs)){
      for(n in 1:length(ys)){
        lines(c(i,i+0.2), c(ys[n], ys[n]), lty = "dashed")
      }
    }
    if(length(ys)!=net$sizes[i]){
      if(i == 1){
        lines(c(i,i), c(ys[5] +  dist, ys[6] -  dist), lty = 21, lwd = 2)
      }else{
        for(m in 1:(length(ys)-1)){
          lines(c(i,i), c(ys[m] + 0.2 * dist, ys[m+1] - 0.2 * dist), lty = 21)
        }
      }
    }
    if(i>=2){
      for(j in 1:length(oldys)){
        for(k in 1:length(ys)){
          lines(c(i-1,i), c(oldys[j], ys[k]), col = "lightgrey")#col.list[[i-1]][j]) 
        }
      }
    }
    if(full & !any(is.na(col.list))){
      if(i>=2){points(rep(i-1, length(oldys)), oldys, cex=3, pch = 20, col = col.list[[i-1]])}
      points(rep(i, length(ys)), ys, cex=3, pch = 20, col = col.list[[i]])
    }else{
      if(is.na(col.list)){
        if(i>=2){points(rep(i-1, length(oldys)), oldys, cex=3, pch = 20)}
        points(rep(i, length(ys)), ys, cex=3, pch = 20)        
      }else{
        if(i>=2){points(rep(i-1, length(oldys)), oldys, cex=3, pch = 20, col = col.list[[i-1]])}
        points(rep(i, length(ys)), ys, cex=3, pch = 20)        
      }
    }
  }
  text(x = 1, y = -1, "Input")
  text(x = length(net$sizes), y = -1, "Output")
  if(n.hidden>1){
    text(x = center.hidden, y = -1, "Hidden Layers")
  }else{
    text(x = center.hidden, y = -1, "Hidden Layer")
  }
}


net.IO.plot <- function(net, data, trueval){
  oldpars       <- par(c("mai", "omi", "mgp", "mar", "oma", "mfrow"))
  newpars       <- oldpars
  newpars$oma   <- c(0,0,1,0)
  newpars$mar   <- c(0,0,1,0)
  newpars$mai   <- c(0,0,0,0)
  newpars$mgp   <- c(0,0,0)
  newpars$mfrow <- c(1,2)
  par(newpars)
  cols        <- colorRampPalette(c("blue", "green"))(200)
  outputs     <- net.IO(net = net, data = data, full = T)
  if(max(data)<=1){
    digitplot.inner(digit = data*255, trueval = trueval, predictval = which.max(outputs[[length(outputs)]])-1) 
  }else{digitplot.inner(digit = data, trueval = trueval, predictval = which.max(outputs[[length(outputs)]])-1)}
  outputcols  <- list()
  for(i in 1:length(outputs)){
    outputcols[[i]]  <-  cols[1+round(outputs[[i]]*199)]
  }
  outputcols[[1]] <- rep(cols[1], length(outputcols[[1]]))
  net.plot.inner(net = net, full = T, full.input = F, col.list = outputcols)
  par(oldpars)
}
