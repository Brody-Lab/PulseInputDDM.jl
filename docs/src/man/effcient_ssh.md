# Efficient Remote SSH

This page is all about how to SSH into spock, della, scotty or any other campus resource _without_ needing to type your password every time.

Edit (or create and edit) an ssh config on your local machine (usually located at ~/.ssh/config) using VIM or your favorite text editor. Add the following code:

```
Host scotty
User [username]
HostName scotty.princeton.edu
ForwardX11 yes
ForwardX11Trusted yes 
```

In this code, you should replace `[username]` with your username (_i.e._ what you log into each server under).

Create RSA keys to facilitate no-password logins [more information here](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2). The steps here are to make the RSA key:

```
>> ssh-keygen -t rsa
```

Then hit enter twice to save the RSA key to the default location and to not include an RSA password. Now add the key to the remote machine via:

```
>> ssh-copy-id [username]@scotty.princeton.edu
```

where again `[username]` is your login name and you can change the address (scotty.princeton.edu) to whichever machine you are trying to access.

With the above code, you can now simply ssh in via

```
>> ssh scotty   
```

and not even have to worry about passwords etc.

These instructions can also be found [here](https://brodylabwiki.princeton.edu/wiki/index.php/Internal:VPN_is_annoying).
