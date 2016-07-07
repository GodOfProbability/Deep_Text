1. The data is taken from Deep Features for Text spotting (http://link.springer.com/chapter/10.1007%2F978-3-319-10593-2_34)
2. The file char_insen.h5 contains char level data (images, 24X24 pixels, greyscale) for the purpose of character insensitive classification.
3. Testing is done on the test set provided by the same paper (contains data from ICDAR and SVT). The test data is stored in test_case_insesitive.h5
4. This data is stored in the position /my_hard_drive/text-data/jaderberg-eccv2014_textspotting/data/
5. weights_word_recognition.hdf5 has the learned weights.
6. Perceptage accuracy: 0.924971142701
7. train_text_non_text_pos.h5 contains the character level case-insensitive dataset for the text-non-text classification
8. It has 185639 postive examples taken from the deep text paper from Jaderburg.
9. Keys are 'X' and 'y'


### What I am doing:
First, collect the negative examples for transfer learning
Second define the architecture for transfer learning
Improve/correct the plotting functions in the classifier.py
Separate the data and plotting related stuff from the main code



Need to write the indices and details of the variable stored in .h5 files.
Also make it more asthetic  pleasing.
Furthermore, each file in a directory should have a short description of it.
All the helping files should either be removed or should be named with _help in the end

