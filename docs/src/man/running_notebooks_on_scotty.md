# Using Julia in a jupyter notebook on scotty or brody-spock

From your local machine, make a ssh connection with scotty, 

```
    >> ssh [username]@scotty.princeton.edu
```

where `[username]` is your scotty username, assuming you have one. If you do not have a username email [PNI help](mailto:pnihelp@princeton.edu). See the [Efficient Remote SSH](@ref) section) about shortcuts to making this quicker.


### Installing the IJulia package

Load the julia module: `$ module load julia/1.2.0`.

Launch Julia by typing `julia`. From here, you need to add the IJulia package so you can use Julia in a notebook. In Julia 1.0, you can access the "package manager" by pressing `]`. From there, enter

```
    (v1.0) pkg > add IJulia
```

After you add the `IJulia` package you have to build it

```
    (v1.0) pkg > build IJulia
```

You might now want to add any packages you use, again in the normal way (for example, you can now add the pulse\_input\_DDM package, as we will do eventually, described in the [Getting the pulse input DDM package from GitHub](@ref) section). You're done (with this part)! You can exit Julia.

```
    > exit()
```

### Edit kernel startup file

I'm not entirely sure why this is necessary, but for some reason, the default setting when opening a jupyter notebook located within a directory that is a Julia module is to make that module the current enviorment. Which can happen since I've included some example notebooks within this repository. But we don't want this to happen, we want the base enviorment to be the enviorment. 

So, in order to tell jupyter to be sure to use the base enviornment whenever if opens a notebook, you have to edit a file located here `/usr/people/[USERNAME]/.local/share/jupyter/kernels/julia-1.2`. The file is called `kernel.json` and you have to delete the lines `"--project=@.",`.

To check that this has taken effect, in your next jupyter notebook, type `] st` in the first cell. It should report something like `Status `/mnt/bucket/people/[USERNAME]/.julia/environments/v1.2/Project.toml`.

### Opening a notebook on scotty and creating a SSH tunnel

Now, that IJulia is all set up, you need to launch a Jupyter notebook on scotty.

First you have to load the anaconda module:

```
    >> module load anacondapy/5.1.0
``` 


In your ssh connection to scotty, type 

```
    >> jupyter notebook --port <port>
``` 

where `<port>` is a 4-digit number that you pick. Once it's running you will be provided a url so that you can launch the notebook in your local browser. But before you can do that, you need to create a "ssh tunnel" from the open port on scotty and a port on your local machine. *In a separate terminal window*, create a *new* ssh connection to scotty, but one that will map the ports, 

```
    >> ssh -L <port>:localhost:<port> [username]@scotty.princeton.edu
``` 

where `<port>` is the port assigned by scotty, and the second `<port>` is one on your local machine (you can pick the same one that scotty did, if you don't happen to be already using it).

Now, copy the url that was created when you launched the jupyter notebook into your local browser, and voila! On a mac, you can press the "command" key (⌘), and the url should become underlined, at which point you can click on it with the  mouse and the link should open in your local browser.

Be sure that when you open your first notebook, you use the Julia 1.2 kernel.

Here is a screen shot of what a typical terminal will look like, showing the url (at the bottom) that needs to be copied and pasted or clicked on:

![notebook-screen-shot](assets/notebook-screen-shot.png)

### Now that everything's set up, how do I open a notebook _again_?

Next time, you only need to:

- SSH into scotty: `$ ssh scotty`.
- Load the anaconda module: `$ module load anacondapy/5.1.0`.
- Load the julia module: `$ module load julia/1.2.0`.
- Launch a jupyter notebook: `$ jupyter notebook --port <port>`.
- Create the ssh tunnel: `$ ssh -L <port>:localhost:<port> scotty`.

These instructions can also be found [here](https://brodylabwiki.princeton.edu/wiki/index.php/Internal:IJulia_notebook_on_scotty).

### Running interactively on spock-brody instead

I'm not sure if this is technically supported by PNI IT or not, but sometimes scotty can be bogged down with lots of jobs. Our personal computing cluster (spock-brody) that PNI manages is mostly set up for batch jobs (like spock) but can be hyjacked to run interactively by executing the following from spock (once you made a SSH connection to it):

```
    $ salloc -p Brody -t 24:00:00 -c 44 srun --pty bash
```

This will request 1 node (44 cores) for 24 hours from the Brody partition. The second bit `srun --pty bash` will open a bash shell on the compute node you just requested. You can now use it like you used scotty above. 

## Jupyter notebook server on spock-brody

If you wanted to run a jupyter notebook server on spock-brody, you need to do some fancy SSH tunneling to get it to work. Here's how to do the whole thing, beginning to end.

First, you have to set a bash enviornment variable so that jupyter doesn't try to write some boring runtime files to a place that you don't have write access to. To do this, modify your `.bashrc` file, which should be located in your home directory. Include the following:

```
    $ unset XDG_RUNTIME_DIR
```

I'm not exactly sure what this does, but I discovered this solution [here](https://github.com/jupyter/notebook/issues/1318).

OK, now we can go ahead and open our jupyter notebook and create our crazy SHH tunnel.

- `ssh scotty`
- `tmux new -s [name] \; split-window -h \; attach`, where `[name]` is whatever you want to name the tmux session.
- In one tmux pane, you are going request the resources you want from spock-brody and create the jupyter notebook, and then in the other, you are going to create a SSH tunnel from the scotty login node to the compute node you are currently using. *Then*, you have to create a SSH tunnel from your local machine to the scotty login node (as we have done before). 

    In detail, in one pane:

    - 
    
        ```
            >> salloc -p Brody -t 24:00:00 -c 44 srun --pty bash
        ```
            
        which will automatically create and "move you" over to a bash shell in the compute node. 
        
    - `module load anacondapy/5.1.0`
    - `module load julia/1.2.0`
    - `jupyter notebook --port <port>`

    Now you should have a jupyter notebook server running in that pane (which is actually a bash shell running on the spock-brody compute node).
    
- In the other pane, you need to create a SSH tunnel from the spock login node (which is where this panel exists) and the spock-brody compute node you are using. The name of that node should have been reported back to you when you ran `salloc`, for example it might be called `spock-brody01-01`. You can view it again by typing `hostname` into the pane. To creat the tunnel, type `ssh -L <port1>:localhost:<port2> [node-name]` where `<port1>` is an open port on the scotty login node, `<port2>` is the port on the spock compute node where your jupyter notebook is running and `[node-name]` is the name of the spock compute node you are using.
- Finally, detach you tmux session `<crtl-D>` and `exit` your SSH connection with scotty. Now, create *another* SSH tunnel from your local machine to the spock login node `ssh -L <port1>:localhost:<port2> scotty` where `<port1>` is an open port on your local machine and `<port2>` is the port you assigned the SSH tunnel to on the scotty login node. After you do this, you will have an SSH connection again with the *scotty login node*. To re-attach your tmux session, type `tmux attach -t [name]`. And then, you can click on the link provided by the jupyter notebook, which should open in a browse on your local machine. Phew!
- In the second tmux pane, you can open an `htop` to see your activity on the compute node `htop -u [username]`.