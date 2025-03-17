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

Activate the virtual environment. Commands vary according to OS and shell (see [the venv documentation](https://docs.python.org/3/library/venv.html)), but in a Windows PowerShell:
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

## Using tacfit

First and foremost: a help message is displayed when running tacfit with the ```-h``` flag:
```
> python -m tacfit -h
```


To use tacfit you need a file of measured time activity curves, say in the file ```tac.txt```
This file is going to be read by numpy.load_txt, so it should follow that format.
If the file is made using [tictac](https://github.com/cwand/tictac) it should work without
problems.

Assume that the time activity curves in ```tac.txt``` are saved under the labels ```input``` and ```tissue```,
and the time label is ```tacq```. We want to fit the ```stepconst``` model (describe below) to the data.
We use the command:
```
> python -m tacfit tac.txt tacq input tissue stepconst --param amp1 0.1 0.05 x --param amp2 0.02 0.0 0.1 --param extent1 10 0 x --leastsq 
```
This will run ```tacfit```, and the result will be shown to the user on the screen.
To explain the options in the command:
* The first three options after ```tacfit``` set the file to be read and the labels of the input function and tissue function.
* The fourth option sets the model to fit.
* The ```--param``` option sets the parameter initial values, lower bound and upper bound. An ```x``` means "no bound".
* The ```--leastsq``` option tells tacfit to fit the model to the data using the least squares method in [lmfit](https://lmfit.github.io/lmfit-py).

Below the various options and settings of ```tacfit``` are explained in detail.


### Models
The models currently implemented in tacfit are
* stepconst
* step2

The list of implemented models and their descriptions can be seen by using the
```--list_models``` option.

#### The stepconst model
In this model, the impulse response function $R(t)$ is\
$$R(t) = \mathrm{amp1}, \quad t < \mathrm{extent1}$$\
$$R(t) = \mathrm{amp2}, \quad t \geq \mathrm{extent1}$$\
This models the situation where the tracer flows through the tissue in some transit time, and
some fraction is extracted by the tissue and stays in the tissue forever.

#### The step2 model
An extension of the ```stepconst``` model, where the impulse response function drops to 0 after some
time $\mathrm{extent2}$:\
$$R(t) = \mathrm{amp1}, \quad t < \mathrm{extent1}$$\
$$R(t) = \mathrm{amp2}, \quad \mathrm{extent1} \leq t < \mathrm{extent2}$$\
$$R(t) = 0, \quad t \geq \mathrm{extent2}$$\
This models a situation where the tracer flows through the tissue in some transit time,
some fraction is extracted by the tissue and stays in the tissue for some time
and then leaves the tissue.

### Setting parameters
Each paramater of the model must be given an initial value and bounds for the fitting procedure.
This is done using the ```--param``` option once for all parameters. For example for the ```stepconst``` model:
```
--param amp1 0.2 0 x --param amp2 0.1 0 x --param extent1 10 x x
```
The ```x``` indicates that no bound should be used. In this example, the parameter ```amp1``` is initialised
to the value ```0.2```, has a minimum value of ```0``` and no upper bound. The parameter ```extent1``` has neither upper nor lower bounds.
All parameters must have an initial value though.

### Exclude data
It is possible to only include a subset of the data by using the ```--tcut N``` option, where
N is an integer describing the number of data points that should be included in the fit.

### Delayed input function
To apply a time delay on the input function use the ```--delay D```, where ```D``` is a float
describing the delay in seconds.

### Save figures to files
As default, figures are shown on the screen. If the figures should instead be saved to a file, use the
```--save_figs PATH``` option, where ```PATH``` is the directory in which the image files should be saved.
