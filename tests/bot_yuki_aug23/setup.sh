sudo apt update

sudo apt install python3
sudo apt install python3-pip

pip3 install --force-reinstall torch torchvision torchaudio
# mps cpu nightly build
#pip3 install --upgrade --no-deps --force-reinstall --pre torch torchvision torchaudio cudatoolkit=11.4.0 --index-url https://download.pytorch.org/whl/nightly/cpu

pip3 install accelerate transformers
