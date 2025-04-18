# tacfit
Fit time-activity curves from dynamic SPECT or PET images to models

## Getting tacfit
Clone the repository to your computer using git:
```
> git clone https://github.com/cwand/tacfit
```

Enter the directory.
Make sure you are on the main branch:
```
> git checkout main
```

Create a new virtual python environment:
```
> python -m venv my_venv
```

Activate the virtual environment. Commands vary according to OS and shell 
(see [the venv documentation](https://docs.python.org/3/library/venv.html)),
but in a Windows PowerShell:
```
> my_venv\Scripts\Activate.ps1
```

Install tacfit and required dependencies
```
> pip install .
```

If everything has gone right, you should be able to run tacfit
```
> python -m tacfit
Starting TACFIT 1.0.1

...
__main__.py: error: the following arguments are required: ...
```

## Non-linear least squares fitting using tacfit

First and foremost: a help message is displayed when running tacfit with 
the ```-h``` flag:
```
> python -m tacfit -h
```


To use tacfit you need a file of measured time activity curves, say in the 
file ```tac.txt```
This file is going to be read by numpy.load_txt, so it should follow that 
format.
If the file is made using [tictac](https://github.com/cwand/tictac) it should 
work without problems.

Assume that the time activity curves in ```tac.txt``` are saved under the 
labels ```input``` and ```tissue```,
and the time label is ```tacq```. We want to fit the ```stepconst``` model 
(describe below) to the data.
We use the command:
```
> python -m tacfit tac.txt tacq input tissue stepconst --param amp1 0.1 0.05 x --param amp2 0.02 0.0 0.1 --param extent1 10 0 x --leastsq 
```
This will run ```tacfit```, and the result will be shown to the user on the 
screen.
To explain the options in the command:
* The first three options after ```tacfit``` set the file to be read and the 
  labels of the input function and tissue function.
* The fourth option sets the model to fit.
* The ```--param``` option sets the parameter initial values, lower bound and 
  upper bound. An ```x``` means "no bound".
* The ```--leastsq``` option tells tacfit to fit the model to the data using
  the least squares method in [lmfit](https://lmfit.github.io/lmfit-py).

Below the various options and settings of ```tacfit``` are explained in detail.


### Models

The list of implemented models and their descriptions can be seen by using the
```--list_models``` option.

#### The stepconst model
In this model, the impulse response function $R(t)$ is\
$$R(t) = a_1, \quad t < t_1$$\
$$R(t) = a_2, \quad t \geq t_1$$\
This models the situation where the tracer flows through the tissue in some 
transit time, and some fraction is extracted by the tissue and stays in the
tissue forever. The parameters are specified in ```tacfit``` as (numerical 
values are just examples):
```
--param amp1 0.1 0.0 0.5 --param amp2 0.01 0.0 0.1 --param extent1 10 0.0 100.0
```


#### The normconst model
Similar to the ```stepconst``` model, except the transition from $a_1$ to 
$a_2$ is smoothed:\
$$R(t) = (a_1 - a_2) (1 - \Phi(t, t_1, w_1)) + a_2,$$\
where $\Phi(t, t_1, w_1)$ is the cumulative normal distribution function 
centered at $t_1$ with standard deviation $w_1$\
$$\Phi(t, t_1, w_1) = \int_{-\infty}^t \frac{1}{\sqrt{2\pi w_1^2}} 
\exp\left( -\frac{(s - t_1)^2}{2w_1^2} \right) ds.$$\
The parameters are specified in ```tacfit``` as (numerical 
values are just examples):
```
--param amp1 0.1 0.0 0.5 --param amp2 0.01 0.0 0.1 --param extent1 10 0.0 100.0 --param width1 2.0 0.0 5.0
```

#### The step2 model
An extension of the ```stepconst``` model, where the impulse response function 
drops to 0 after some time $\mathrm{extent2}$:\
$$R(t) = a_1, \quad t < t_1$$\
$$R(t) = a_2, \quad t_1 \leq t < t_2$$\
$$R(t) = 0, \quad t \geq t_2$$\
This models a situation where the tracer flows through the tissue in some 
transit time, some fraction is extracted by the tissue and stays in the tissue 
for some time and then leaves the tissue.
The parameters are specified in ```tacfit``` as (numerical values are just 
examples):
```
--param amp1 0.1 0.0 0.5 --param amp2 0.01 0.0 0.1 --param extent1 10 0.0 100.0 --param extent2 100 0.0 500
```

#### The stepnorm model
Similar to the ```step2``` model, except the transition from $a_2$ to 0 is
smoothed similarly to the ```normconst``` model:\
$$R(t) = (a_1 - a_2) (1 - H(t - t_1)) + a_2 (1 - \Phi(t, t_2, w_2)),$$\
where $H(t)$ is the Heaviside step function.
The parameters are specified in ```tacfit``` as (numerical values are just 
examples):
```
--param amp1 0.1 0.0 0.5 --param amp2 0.01 0.0 0.1 --param extent1 10 0.0 100.0 --param extent2 100 0.0 500 --param width2 10 0 100
```

#### The norm2 model
Similar to the ```step2``` and ```stepnorm``` model, except both transitions
are smoothed:\
$$R(t) = (a_1 - a_2) (1 - \Phi(t, t_1, w_1)) + a_2 (1 - \Phi(t, t_2, w_2)),$$\
The parameters are specified in ```tacfit``` as (numerical values are just 
examples):
```
--param amp1 0.1 0.0 0.5 --param amp2 0.01 0.0 0.1 --param extent1 10 0.0 100.0 --param extent2 100 0.0 500 --param width1 2 0 10 --param width2 10 0 100
```

### Setting parameters
Each fitting parameter of the model must be given an initial value and bounds
for the fitting procedure.
This is done using the ```--param``` option once for all parameters. For 
example for the ```stepconst``` model:
```
--param amp1 0.2 0 x --param amp2 0.1 0 x --param extent1 10 x x
```
The ```x``` indicates that no bound should be used. In this example, the 
parameter ```amp1``` is initialised to the value ```0.2```, has a minimum value
of ```0``` and no upper bound. The parameter ```extent1``` has neither upper 
nor lower bounds.
All parameters must have an initial value though.

It is also possible to add "fixed" parameters to the model, i.e. parameters
that should not be varied during the fit. To do this, use the ```--exprparam```
option:
```
--param amp1 0.2 0 x --exprparam amp2 "0.1 * amp1" --exprparam extent1 10.0
```
This will fit only one parameter of the ```stepconst``` model (```amp1```), 
while ```amp2``` is kept fixed at 10% of ```amp1``` and ```extent1``` is set
to the value 10.0.

### Exclude data
It is possible to only include a subset of the data by using the 
```--tcut N``` option, where N is an integer describing the number of data 
points that should be included in the fit, or the ```--scut T``` option, where
T is the time of the last data point to include.

It is possible to scan over a range of different tcut-values, by giving more
than one value: ```--tcut N1 N2 ...``` (this feature is not yet implemented
for Monte Carlo sampling). In this case, a plot is shown after
the fit with the optimal parameter values for each tcut. To hide the progress
bar, use the option ```--hideprogress``` (useful if piping output to a file).



### Delayed input function
To apply a fixed time delay on the input function use the ```--delay D```, 
where ```D``` is a float describing the delay in seconds. Alternatively the
input function delay can be treated as a free fitting parameter. To do this,
add an extra parameter like this:
```--param _delay 1.5 0.0 10.0``` (numerical values are examples, and an 
```x``` can also be used here to specify no bounds.)

### Save figures to files
As default, figures are shown on the screen. If the figures should instead be 
saved to a file, use the
```--save_figs PATH``` option, where ```PATH``` is the directory in which the 
image files should be saved.


## Monte Carlo sampling of the posterior parameter probability distribution

To check whether the uncertainties on the parameter estimates found by 
```lmfit``` is adequate, it is possible to
do a Monte Carlo sampling of the posterior parameter likelihood function.
Let us first set up the notation. We write the time points of the measurements 
(the acquisition time of each image in the series)
as $t_i, 1 \leq i \leq n$. The measured tissue activity concentrations are then
$y_i = y(t_i)$, and the measured input function
is $c(t)$. The parameters (to be sampled) of the model function is 
$\beta_j, 1\leq j \leq k$.
Now, given a choice of $\beta_j$, the likelihood that we measure a single 
tissue activity concentration $y_i$ is given by\
$$\mathcal{L}(y_i | c(t), \beta, \sigma_i) = 
\frac{1}{\sqrt{2\pi\sigma_i^2}}
\exp\left(-\frac{(y_i - \mathcal{M}_i(c(t), \beta))^2}{2\sigma_i^2}\right)$$,\
where $\mathcal{M}_i$ is the modeled tissue activity curve evaluated at time 
point $t_i$, $\sigma_i$ is the estimated uncertainty on $y_i$ and we have 
assumed $y_i$ is drawn from a normal distribution with mean $\mathcal{M}_i$
and variance $\sigma_i^2$.
The uncertainties $\sigma_i$ will be treated as nuisance parameters of the 
model, since they are rarely estimated directly.
The combined likelihood of measuring all $y_i$ is then (assuming independent 
measurements):\
$$\mathcal{L}(y | c(t), \beta, \sigma) = 
\prod_i \mathcal{L}(y_i | c(t), \beta, \sigma_i) = 
\prod_i \frac{1}{\sqrt{2\pi\sigma_i^2}}
\exp\left(-\frac{(y_i - \mathcal{M}_i(c(t), \beta))^2}{2\sigma_i^2}\right)$$.

Using Bayes' theorem, we can now write the likelihood function for the 
parameters $\beta$ and $\sigma$:\
$$\mathcal{L}(\beta, \sigma | y, c(t)) = 
\frac{\mathcal{L}(\beta, \sigma) 
\mathcal{L}(y | c(t), \beta, \sigma)}{\mathcal{L}(y)} = 
\frac{1}{Z} \mathcal{L}(\beta, \sigma) 
\prod_i \frac{1}{\sqrt{2\pi\sigma_i^2}}
\exp\left(-\frac{(y_i - \mathcal{M}_i(c(t), \beta))^2}{2\sigma_i^2}\right) $$.\
The marginal likelihood $\mathcal{L}(y)$ is independent on the model parameters 
and can be identified as the partition function $Z$, the calculataion of which 
is handled implicitly by the Monte Carlo sampling algorithm. 
Here we assume the input function $c(t)$ is a fixed parameter of the model. 
Methods to include the uncertainty on the measurements of $c(t_i)$ are under 
development.



To sample this likelihood function in ```tacfit``` we use the package
[emcee](https://emcee.readthedocs.io/en/stable/). To run the sampling we need 
to specify what model to use for $\sigma_i$. At this point, two different 
models are implemented:
* ```const```: All uncertainties are equal, i.e. $\sigma_i = \sigma \forall i$.
* ```sqrt```: Uncertainties scale with the square root of the measured data, 
  i.e. $\sigma_i^2 = \sigma^2 y_i$.

The sampling is run in ```tacfit``` very similarly to a least squares fit, 
albeit with a handful more options:
```
> python -m tacfit tac.txt tacq input tissue stepconst --mcpost --mc_steps 1000 --mc_walkers 300 --mc_error const --mc_threads 4 --mc_burn 300 --mc_thin 30 --rng_seed 42 --param amp1 0.1 0.0 0.3 --param amp2 0.02 0.0 0.1 --param extent1 10 0 100 --param sigma 100 0.1 10000
```
The new options are:
* ```--mcpost```: Tells ```tacfit``` to run the Monte Carlo sampling algorithm.
* ```--mc_steps```: How many updating steps the algorithm will make.
* ```--mc_walkers```: The number of independent walkers. See the 
  [emcee documentation](https://emcee.readthedocs.io/en/stable/) for more info.
* ```--mc_burn```: How many samples to discard as burn-in.
* ```--mc_thin```: Only keep one out of every N samples (used to avoid 
  autocorrelation).
* ```--mc_error```: The error model to use.
* ```--mc_threads```: The number of threads to start
* ```--hideprogress```: Hides the progress bar, useful if output is piped to 
  a file.
* ```--rng_seed```: Sets the RNG-seed, to be able to reproduce results.

A couple of changes are also in effect for the parameter specification:
* The use of an ```x``` to specify "no bound" is not allowed (since the prior 
  probability for each parameter must be non-zero).
* A new parameter ```_sigma``` must be specified (with interpretation based on 
  the error model). The underscore
  indicates that this is a nuisance parameter (as opposed to some other sigma, 
  which may be a model parameter).
* To model input function delay as a nuisance parameter, add the parameter 
  ```_delay``` to the list with appropriate initial value guess and bounds.

