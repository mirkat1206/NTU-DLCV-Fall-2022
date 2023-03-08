python3 -c "import clip; clip.load('ViT-B/32')"
python3 -c "import language_evaluation; language_evaluation.download('coco')"
python3 -c "import timm; timm.create_model('vit_huge_patch14_224_clip_laion2b', pretrained=True)"

wget --no-check-certificate 'https://www.dropbox.com/s/xtd2if7leb9j8qg/hw3_bestcheckpoint.tgz?dl=0' -O bestcheckpoint.tgz
tar xvf bestcheckpoint.tgz