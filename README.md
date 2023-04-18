# safedriveCV

## Getting started
There are a few things you'll need to install in order to get this project up and running, don't worry these steps are simple and easy to follow. Thankfully you are all using the same OS so that makes this signficantly easier to explain.

1. Install Xcode

Xcode is a package on MacOS that includes a lot of developer tools you're probably going to need. Open a terminal and run the following command to get it installed: `xcode-select --install`

Xcode includes a package called cURL which will let you install the next important package.

2. Install homebrew

Homebrew is a package manager, lets you install other things very easily. Again go to your terminal, this time run this command
`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

now we can get to the things needed specifically for this project

3. Install pyenv

pyenv is a python version manager. installing python versions on macos is known to be a pain in the ass, [pyenv](https://github.com/pyenv/pyenv) makes the process much less annoying.
Now that you have homebrew, installing pyenv will be super simple, just run `brew install pyenv`

Next you need to figure out what kind of shell your computer is running. Not super important to understand what a shell is but its basically the language your terminal uses. newer macs use ZSH and older ones usually use BASH, im not sure when the cutoff is but to determine what yours is using just run `which $SHELL`.

if the command prints `/bin/zsh/` you're using zsh, if it prints `/bin/bash/`, bash. 

depending on what your terminal prints you will either run:
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
```
for bash, or:
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```
for zsh.

this makes sure that pyenv will start up everytime you open a new terminal window. fully quit your terminal and reopen so the changes take effect.

4. installing Python

Now that we have pyenv installing python will be super easy.

in this project im using python version 3.9.9. its pretty new while not being super broken so it strikes a good balance.
to install it, run the command `pyenv install 3.9.9`. Once the installation is finished, run the command `pyenv global 3.9.9`.
now run `python --version`, you should get `Python 3.9.9`. if not try closing your terminal and reopening. still broken? idk man good luck

5. installing Pipenv

This is the final installation step. [Pipenv](https://docs.pipenv.org/) creates a virtual environment that makes tracking all the python libraries you need super easy, it also makes it easy for you guys to make sure you're using all the same library versions as the ones used in this project. 

To get pipenv just run `pip install pipenv`. 

6. Clone this project. 

Before you download this project onto you computer, I reccommend you create a folder. I'm not going to bore you with terminal commands, just tell you enough to get you to a place you can download this to. 

`ls` - this lists all files in the folder you are currently in.

`pwd` - prints the folder you are currently in `/Users/your-name`

`mkdir` - make a folder

`cd 'foldername'` - move into that folder or `cd ..` to move back one directory cd `../../` to move back two, etc.

copy and paste the following list of commands all at once 
```
mkdir DriveMate
cd DriveMate
git clone https://github.com/ijs32/safedriveCV.git
```
if you now run the command `pwd` you will see you are in the directory `/Users/your-name/DriveMate`
if you run `ls` you will see a new folder called `savedriveCV`, this is the project folder.

now: 
```
cd savedriveCV
pipenv shell
pipenv install
```

and you're done, you are now in the project folder and everything is installed. everytime you enter the project remember to run `pipenv shell`, this will start up the virtual environment, without running this you wont have access to the libraries you installed since you are outside the virtual environment they are installed in.

## Suggestions

I would suggest opening your browser and searching for VScode. its the easiest and most used code editor. Once its installed, open it and press `Command + Shift + P` this will open a little search bar at the top, type command `Shell command` and select the install option. Now, when you open your terminal and `cd DriveMate/safedriveCV` to this project, you can type the command `code .` to open the folder in VScode. Its good to learn to use the terminal because you'll be using it to run any files with `python filename.py`.
