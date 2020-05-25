# Project Title

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members: 
- Austin Le, aule@ucsd.edu
- Justin Kang, jjk139@ucsd.edu 
- Sravya Voleti, srvoleti@ucsd.edu
- Sruthi Vedantham, srvedant@ucsd.edu
- Zahra Masood, zbmasood@ucsd.edu

## Abstract

(10 points) 

Our concept for this generative art project is to generate lyrics for a new Beyonce song by training a RNN model using a variety of Beyonce lyrics. We hope to extract features from these newly generated lyrics and compare them to features in already existing Beyonce lyrics to measure how accurately our model was able to create this generative art. We will be generating our song lyrics using a character-based RNN. This topic is interesting because Beyonce is one of the most popular artists from our generation. She has sold over 100 million records worldwide, and is known as one of the world’s best selling music artists. As such an influential and popular person in the music industry, we felt it would be interesting to examine the elements of her song lyrics and see if we could generate an accurate representation of her writing style. In lecture, we learned a lot about generative art and learning with neural networks. We investigated various forms of generative text, generative images, and generative sounds. 

This project is an extension of these topics as we’re using a character-based RNN to create generative text (lyrics) of a particular artist, Beyonce and quantitatively comparing the similarity of it to song lyrics that already exist (https://github.com/roberttwomey/dsc160-code/blob/master/examples/text-generation-rnn.ipynb). Some of the libraries we will be using are TensorFlow and other basic libraries such as Pandas and Numpy. We will start off by scraping the website that contains all of Beyonce’s albums and creating a dataframe that contains the name of each song, the album it’s from, the total number of words, and the lyrics for that song. Then we will go on to vectorize the text by mapping the string representations to a numerical one. After this, we will begin to create training examples so our model can use our input in order to predict what words will come next based on a given sequence of words. Then, we will go on to actually build our RNN model with three layers: the input layer, the LSTM layer, and the output layer. After this, we will use our training examples to train our model and see how well it performs when given sequences of words. After optimizing our model based on how well it performs, we will finally go on to actually generating our text using a prediction loop. After generating our new lyrics, we will compare features calculated on the original training data with features calculated on the newly generated lyrics. These features include the number of words in the lyrics, common phrases, and common words.The training data we will be using are all of Beyonce’s existing song lyrics. We will obtain this data by scraping an online database that contains the lyrics to all of Beyonce’s songs (https://www.azlyrics.com/k/knowles.html).

We hope that our system will produce the lyrics to the ultimate Beyonce song song that is based on her existing body of work. We will calculate similarity by comparing features we calculate on her existing body of work with features we calculate on the newly generated song lyrics. Using a form of generative art, we are planning to take the songs of Beyonce and use all the lyrics from her songs in order to generate lyrics to a new song. We will present the newly created song lyrics that we generated and compare it to her previous works to see if the new lyrics make sense and can add a different element. Our output would be in a text format. One specific challenge for our project would be finding the right database that contains the necessary song lyrics for the Beyonce songs we want to use to generate our own lyrics. Copyright issues as well as being able to scrape the necessary song metadata (ex. Album year, song year) could force our group to manually acquire the necessary lyrics from various sources instead of one single source. Generating a coherent lyric once we create a functional model would become another potential setback to our project. Since our group has no real experience using neural networks, how we are able to correctly implement models could significantly affect the performance of our predicted lyrics. Some work we will be referencing to complete our project includes: Professor Twomey’s RNN and generative text example: https://github.com/roberttwomey/dsc160-code/blob/master/examples/text-generation-rnn.ipynb, Exploring the unreasonable effectiveness of RNNs: http://karpathy.github.io/2015/05/21/rnn-effectiveness/, Utilizing LSTMs and CNNs for creation of song lyrics: https://techxplore.com/news/2019-01-song-lyrics-style-specific-artists.html, and Creating word-based RNNs using tensorflow: https://github.com/hunkim/word-rnn-tensorflow

## Data and Model

(10 points) 

In the final submission, this section will describe both the data you use for this project and any pre-existing models/neural nets. For each you should provide the name, a textual description, and a link. If there is a paper (for neural net) link that as well.
- Such and such Neural Net. The short description of this neural net. 
  - [link to code]().
  - [Title of Paper with Link](). 
- Training data. Short description of training data including bibliographic info. [link to data]().

## Code

(20 points)

This section will link to the various code for your project (stored within this repository). Your code should be executable on datahub, should we choose to replicate your result. This includes code for: 

- code for data acquisition/scraping
- code for preprocessing
- training code (if appropriate)
- generative methods

Link each of these items to your .ipynb or .py files within this seection, and provide a brief explanation of what the code does. Reading this section we should have a sense of how to run your code.

## Results

(30 points) 

This section should summarize your results and will embed links to documentation to significant outputs. This should document both process and show artistic results. This can include figures, sound files, videos, bitmaps, as appropriate to your generative art idea. Each result should include a brief textual description, and all should be listed below: 

- image files (`.jpg`, `.png` or whatever else is appropriate)
- audio files (`.wav`, `.mp3`)
- written text as `.pdf`

## Discussion

(30 points, three to five paragraphs)

The first paragraph should be a short summary describing your results.

The subsequent paragraphs could address questions including:
- Why is this culturally innovative?
- How does your generative computational approach differ from traditional art/music/cultural production? 
- How do your results relate to broader social, cultural, economic political, etc., issues? 
- What are the ethical concerns for this form of generative art? 
- In what future directions could you expand this work?

## Team Roles

Provide an account of individual members and their efforts/contributions to the specific tasks you accomplished.

## Technical Notes and Dependencies

Any implementation details or notes we need to repeat your work. 
- Additional libraries you are using for this project
- Does this code require other pip packages, software, etc?
- Does this code need to run on some other (non-datahub) platform? (CoLab, etc.)

## Reference

All references to papers, techniques, previous work, repositories you used should be collected at the bottom:
- Papers
- Repositories
- Blog posts
