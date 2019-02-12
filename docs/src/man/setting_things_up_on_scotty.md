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

- SSH into scotty: `$ ssh scotty`.
- Load the anaconda module: `$ module load anacondapy/5.1.0`.
- Activate the julia environment: `$ source activate julia`.
- Launch a jupyter notebook: `$ jupyter notebook`.
- Create the ssh tunnel: `$ ssh -L <port>:localhost:<port> scotty`.

These instructions can also be found [here](https://brodylabwiki.princeton.edu/wiki/index.php/Internal:IJulia_notebook_on_scotty).

## Running interactively on spock-brody instead

I'm not sure if this is technically supported by PNI IT or not, but sometimes scotty can be bogged down with lots of jobs. Our personal computing cluster (spock-brody) that PNI manages is mostly set up for batch jobs (like spock) but can be hyjacked to run interactively by executing the following from spock (once you made a SSH connection to it):

```
    $ salloc -p Brody -t 24:00:00 -c 44 srun --pty bash
```

This will request 1 node (44 cores) for 24 hours from the Brody partition. The second bit `srun --pty bash` will open a bash shell on the compute node you just requested. You can now use it like you used scotty above. 

## Jupyter notebook server on spock-brody

If you wanted to run a jupyter notebook server on spock-brody, you need to do some fancy SSH tunneling to get it to work. Here's how to do the whole thing, beginning to end.

First, you need to create a file and modify some jupyter defaults. Without doing this, jupyter tries to write some files in a location that you won't have permissions for. 

- Next, you need to create a "cookie secret" file (I have no idea what this is, but it sounds delicious!) You can find details about this [here](https://jupyterhub.readthedocs.io/en/stable/getting-started/security-basics.html#cookie-secret). `$ openssl rand -hex 32 > [path]/jupyterhub_cookie_secret`. Here `[path]` is something you decide and have write access to. For example I used `/mnt/bucket/people/briandd/.jupyter/`.
- Next, you need to create a default jupyter config file by typing `$ jupyter notebook --generate-config`. I learned about this [here](https://jupyter-notebook.readthedocs.io/en/stable/config.html). Now you need to edit this file to indicate the location of the cookie secret you just made. Modify the lines in the newly created file `jupyter_notebook_config.py`, which is located in `$HOME/.jupyter` so that it reads

```
    c.Notebook.cookie_secret_file = '[path]/jupyter_cookie_secret'
```

OK, now we can go ahead and open our jupyter notebook and create our crazy SHH tunnel.

- `ssh spock`
- `tmux new -s [name] \; split-window -h \; attach`, where `[name]` is whatever you want to name the tmux session.
- In one tmux pane, you are going request the resources you want from spock-brody and create the jupyter notebook, and then in the other, you are going to create a SSH tunnel from the spock login node to the compute node you are currently using. *Then*, you have to create a SSH tunnel from your local machine to the spock login node (as we have done before). 

    In detail, in one pane:

    - 
    
        ```
            >> salloc -p Brody -t 24:00:00 -c 44 srun --pty bash
        ```
            
        which will automatically create and "move you" over to a bash shell in the compute node. 
        
    - `module load anacondapy/5.1.0`
    - `source activate julia`
    - Each time you try to do this, you have to `$ unset XDG_RUNTIME_DIR`. I'm not exactly sure what this does, but I discovered this solution [here](https://github.com/jupyter/notebook/issues/1318).
    - `jupyter notebook`

    Now you should have a jupyter notebook server running in that pane (which is actually a bash shell running on the spock-brody compute node).
    
- In the other pane, you need to create a SSH tunnel from the spock login node (which is where this panel exists) and the spock-brody compute node you are using. The name of that node should have been reported back to you when you ran `salloc`, for example it might be called `spock-brody01-01`. To creat the tunnel, type `ssh -L <port1>:localhost:<port2> [node-name]` where `<port1>` is an open port on the spock login node, `<port2>` is the port on the spock compute node where your jupyter notebook is running and `[node-name]` is the name of the spock compute node you are using.
- Finally, detach you tmux session `<crtl-D>` and `exit` your SSH connection with spock. Now, create *another* SSH tunnel from your local machine to the spock login node `ssh -L <port1>:localhost:<port2> spock` where `<port1>` is an open port on your local machine and `<port2>` is the port you assigned the SSH tunnel to on the spock login node. After you do this, you will have an SSH connection again with the *spock login node*. To re-attach your tmux session, type `tmux attach -t [name]`. And then, you can clikc on the link provided by the jupyter notebook, which should open in a browse on your local machine. Phew!
- In the second tmux pane, you can open an `htop` to see your activity on the compute node `htop -u [username]`.