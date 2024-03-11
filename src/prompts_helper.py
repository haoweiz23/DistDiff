PHOTO_PROMPTS = {
	'imagenet': "A photo of a {}.",
	'imagenet-sketch': "A black and white pencil sketch of a {}.",
	'cifar10': "A photo of a {}.",
	'cifar100': "A photo of a {}.",
	'cifar100_subset': "A photo of a {}.",
	'birdsnap': "A photo of a {}, a type of bird.",
	'country211': "A photo I took in {}",
	'cub': "A photo of a {}, a type of bird.",
	'caltech-101': "A photo of a {}.",
	'caltech256': "A photo of a {}.",        
	'oxford_pets': "A photo of a pet {}.",
	'stanford_cars': "A photo of a {} car.",
	'oxford_flowers': "A photo of a {}, a type of flower.",
	'food101': "A photo of a {}, a type of food.",
	'fgvc_aircraft': "A photo of a {}, a type of aircraft",
	'sun397': "A photo of a {}.",
	'dtd': "{} texture.",
	'eurosat': "A centered satellite photo of {}.",
	'ucf101': "A photo of a person doing {}."
}

def return_photo_prompts(dataset):
    return PHOTO_PROMPTS[dataset]