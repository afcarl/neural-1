# neural
A collections of some fun toy examples

The neural net code snippets were written by Alec Radford [here](https://github.com/Newmu/Theano-Tutorials)

1. Getting error bars in neural network predictions by some hacky ways, note that the prediction is too confident away from the training data and I would strongly argue that this is not the correct way to get predict error bars

* for reference, this is the prediction made using only one single set of weights after training
  ![dropout](https://github.com/thangbui/neural/blob/master/temp/sgd_one.png)

* using dropout, i.e. after training, make prediction as normal but with dropout, then average over the predictions made
  ![dropout](https://github.com/thangbui/neural/blob/master/temp/dropout.png)

* using weights from last few SGD runs, i.e. after training, run SGD for a few more iterations and make prediction after each run, then average over the predictions made
  ![dropout](https://github.com/thangbui/neural/blob/master/temp/sgd.png)

* using weights from last few SGD runs with some gaps in between, i.e. after training, run SGD for a few more iterations and make prediction after every 10 runs, then average over the predictions made
  ![dropout](https://github.com/thangbui/neural/blob/master/temp/sgd_gap.png)

2. Getting error bars using an alternative method, GPs. Here I use GPs with zero mean and SE kernel -- the prediction looks more promising and much much better than the previous tricks

* one GP
 ![gp](https://github.com/thangbui/neural/blob/master/temp/gp_one.png)

* stacked GPs
 ![gp](https://github.com/thangbui/neural/blob/master/temp/gp_two.png)
