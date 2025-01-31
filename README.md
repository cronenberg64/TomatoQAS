# TomatoQAS
I'm making a tomato quality assessment system (TomatoQAS) as a part of my university's Project Based Learning course. I have also attached some documentation in the form of a poster that was used for my presentation during the end of my course that may help with the context of this project as well as some information or legitimacy regarding the project.

To recreate this model on your own, you can simply download the VGG16.py, test_dir.py, and test_img.py files and after downloading the datasets from the given data repository, you can start training the model. To do that, you'll have to create a venv in the directory of choice and then run VGG16.py to initialize the training procedure (make sure to rename the directory for training and validation as my path will differ from yours). After the training is done, you can run test_dir.py to try the validation dataset (also adjust the directory manually in the code). The test_dir file is specifically used to assess the model's accuracy or F1 score for directories and not singular files, to assess an individual image, you can run the test_img file after you have adjusted the path in the code manually.

That should cover all the essential information, if there are any inquiries please don't hesitate to contact me, thank you!

Some important notes: 
The datasets required to train, validate, and test the model developed is stored on kaggle at the following link:
https://www.kaggle.com/datasets/cronenberg64/tomatoqas-dataset



