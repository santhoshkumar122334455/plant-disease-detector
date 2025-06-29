import gdown

url = 'https://drive.google.com/file/d/1e0xmtz08OXyHWf5fjQbJSgxE_ABfE_G6/view?usp=sharing'
output = 'efficientnet_checkpoint.keras'
gdown.download(url, output, quiet=False)
