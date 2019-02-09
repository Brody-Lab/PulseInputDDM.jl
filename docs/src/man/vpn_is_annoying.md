# Efficient Remote SSH

(Thanks to Nick Roy of the PillowLab for these instructions!)

This page is all about how to ssh into spock, della, scotty or any other campus resource without needing to use VPN.

The basic idea here is to use RSA to ssh through an intermediate, open, server. In this case, the two options are nobel (nobel.princeton.edu) and arizona (arizona.princeton.edu). This should work on any unix machine (linux and MAC). Windows people should seek immediate attention. The steps are as follows:

Edit (or create and edit) an ssh config on your local machine (usually located at ~/.ssh/config), and add the following code:

```
   Host nobel
   User UNAME
   HostName nobel.princeton.edu
   Host spock
   User UNAME
   HostName scotty.princeton.edu
   ForwardX11 yes
   ForwardX11Trusted yes
   ProxyCommand ssh nobel -oClearAllForwardings=yes -W %h:%p   
```

In this code, you should replace `UNAME` with your username (_i.e._ what you log into each server under) and nobel can be replaced everywhere with arizona if you would like to use arizona as the pass-through server. To access other machines, replace spock with della or scotty.

Create RSA keys to facilitate no-password logins [more information here](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2). The steps here are to make the RSA key:

```
    >> ssh-keygen -t rsa
```

Then hit enter twice to save the RSA key to the default location and to not include an RSA password. Now add the key to the pass-through server and the remote machine via:

```
    >> ssh-copy-id UNAME@nobel.princeton.edu
    >> ssh-copy-id UNAME@spock.princeton.edu
```

where again UNAME is your login name and you can change the address (spock.princeton.edu) to whichever machine you are trying to access. Make sure to do this first for either arizona or nobel (whichever you decide to use) and then again for the machine you are trying to access.

With the above code, you can now simply ssh in via

```
    >> ssh scotty   
```

and not even have to worry about VPN, passwords etc.

These instructions can also be found [here](https://brodylabwiki.princeton.edu/wiki/index.php/Internal:VPN_is_annoying).