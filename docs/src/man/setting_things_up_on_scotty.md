# Preliminaries to fitting the model interactively on scotty

First you need to install Julia in your scotty user directory via anaconda. This will be a *whole new installation of Julia* (not the installation you might normally use on scotty if you typed `module load julia`) which means if you've used another installation of Julia on scotty before, all the packages you might have added won't be there. Using a new anaconda installation of Julia allows easier use of Jupyter notebooks via a SSH tunnel. 

## Installing Julia via anaconda

From your local machine, make a ssh connection with scotty, 

```
    >> ssh [username]@scotty.princeton.edu
```

where `[username]` is your scotty username, assuming you have one. If you do not have a username email [PNI help](mailto:pnihelp@princeton.edu).

(If you haven't configured your local machine so that you can avoid using the VPN to access scotty, do that first. Instructions are provided on the [Efficient Remote SSH](@ref) page.)

Next load an anaconda module on scotty (for example, 5.1.0), 

```
    >> module load anacondapy/5.1.0
``` 

Presently, you are using the "base" anaconda environment, for which PNI users do not have write privileges. You need to create a new environment, which we will call "Julia",

```
    >> conda create --name julia
``` 

Now that you've created this new environment, you need to "activate" it, 

```
    >> source activate julia
``` 

Next, you need to install Julia within this environment via [anaconda](https://anaconda.org/conda-forge/julia), 

```
    >> conda install -c conda-forge julia
```

You are using a *whole new version of Julia* (_not_ the one you use when you execute `module load julia` on scotty or spock), so you will need to re-install lots of packages (if you have used Julia on scotty or spock before), most specifically, the IJulia package so you can use Julia in a Jupyter notebook.

## Installing the IJulia package

Once, Julia is installed, you can launch it by typing `julia`. From here, you need to add the IJulia package so you can use Julia in a notebook. In Julia 1.0, you can access the "package manager" by pressing `]`. From there, enter

```
    (v1.0) pkg > add IJulia
```

You might now want to add any packages you use, again in the normal way (for example, you can now add the pulse\_input\_DDM package, as we will do eventually, described in the [Getting the pulse input DDM package from GitHub](@ref) section). You're done (with this part)! You can exit Julia.

## Opening a notebook on scotty and creating a SSH tunnel

Now, that Julia is all set up, you need to launch a Jupyter notebook on scotty (which is possible because you're already using the anaconda module, which contains jupyter). In your ssh connection to scotty, type 

```
    >> jupyter notebook
``` 

Once this has executed, it will pick a port on which to run and provide you a url so that you can launch the notebook in your local browser. But before you can do that, you need to create a "ssh tunnel" from the open port on scotty and a port on your local machine. *In a separate terminal window*, create a *new* ssh connection to scotty, but one that will map the ports, 

```
    >> ssh -L <port>:localhost:<port> [username]@scotty.princeton.edu
``` 

where `<port>` is the port assigned by scotty, and the second `<port>` is one on your local machine (you can pick the same one that scotty did, if you don't happen to be already using it).

Now, copy the url that was created when you launched the jupyter notebook into your local browser, and voila! On a mac, you can press the "command" key (⌘), and the url should become underlined, at which point you can click on it with the  mouse and the link should open in your local browser.

Here is a screen shot of what a typical terminal will look like, showing the url (at the bottom) that needs to be copied and pasted or clicked on:

![notebook-screen-shot](assets/notebook-screen-shot.png)

## Now that everything's set up, how do I open a notebook _again_?

Next time, you only need to:

- SSH into scotty: `>> ssh scotty`.
- Load the anaconda module: `>> module load anacondapy/5.1.0`.
- Activate the julia environment: `>> source activate julia`.
- Launch a jupyter notebook: `>> jupyter notebook`.
- Create the ssh tunnel: `>> ssh -L <port>:localhost:<port> scotty`.

These instructions can also be found [here](https://brodylabwiki.princeton.edu/wiki/index.php/Internal:IJulia_notebook_on_scotty).

## Running interactively on spock-brody instead

I'm not sure if this is technically supported by PNI IT or not, but sometimes scotty can be bogged down with lots of jobs. Our personal computing cluster that PNI manages is mostly set up for batch jobs (like spock) but can be hyjacked to run interactively by executing the following from spock (once you made a SSH connection to it):

```
    >> salloc -p Brody -t 24:00:00 -c 44 srun --pty bash
```

This will request 1 node (44 cores) for 24 hours from the Brody partition. The second bit `srun --pty bash` will open a bash shell on the compute node you just requested. You can now use it like you used scotty above. 