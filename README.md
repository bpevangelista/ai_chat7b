# ai_chat7b

### generating keys
ssh-keygen -t ed25519 -C "your@email.com"

### ssh .bashrc setup
eval $(ssh-agent -s)
ssh-add ~/.ssh/git_ed25519 

### aws config
[default]
region=us-east
