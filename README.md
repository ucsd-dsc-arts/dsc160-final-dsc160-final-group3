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

### Data
Our data for our final project was constructed primarily through web scraping lyrics for Beyonce songs. The source of our lyrics came from azlyrics.com (https://www.azlyrics.com/k/knowles.html) and where we were able to access close to 200 Beyonce songs. The final dataset consisted of songs that were found in albums created by Beyonce. As a result of this restriction and our exploratory data analysis, we ended up using approximately 127 songs for our final dataset. The final dataset took into account duplicate songs and non-album specific songs Beyonce was associated with (ie. The Lion King soundtrack). We were able to create two separate data sources once our main dataset was created. The first data source (https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group3/blob/master/data/lyrics_data_v2.csv) was a dataframe that had one column containing the title of the song and another column that had the correct lyric based on the correct song title it was associated with. Our second data source (https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group3/blob/master/data/lyrics_text.txt) was a txt file that contained just the lyrics of all the songs as one large string. For cleaning purposes, items such as newlines (“\n”) and carriage returns (“\r”) were removed prior to the data sources being created, but contractions found in the lyrics were maintained.

### Model
- Used a Recurrent Neural Net (RNN). Our sources:
  - Link to code: https://github.com/roberttwomey/dsc160-code/blob/master/examples/text-generation-rnn.ipynb
  - Link to paper: https://www.tensorflow.org/tutorials/text/text_generation
  
The model we used was a character-based Recurrent Neural Network (RNN) to generate the text. To begin with, we read our text file containing the lyrics in and then started processing the text. To do this, we vectorized the text and mapped the strings to a numerical representation to have an integer representation for each character. We then divided the text into example sequences of a certain size to create the training batches and then shuffled these batches. After this is when we actually began to build our model. We utilized different functions from the tensorflow package, such as tf.keras.Sequential (https://www.tensorflow.org/api_docs/python/tf/keras/Sequential). Then we ran and examined our model in order to see the shape of our output as well as other parameters. At this point is when we began to actually train our model using an optimizer and a loss function. Here we made the model predict the class of the next character and would see if this prediction was right or wrong. After this is when we began to execute this training that we just completed by choosing the number of Epochs to use in order to train. We experimented with different numbers of epochs to appropriately train the model. Finally, our last step was to actually generate the text. To do this, we implemented a prediction loop which initialized the RNN state, got the prediction distribution, and used a categorical distribution to predict the next character. We then used this prediction loop function in order to generate text based on a particular starting string.

## Code

(20 points)

This section will link to the various code for your project (stored within this repository). Your code should be executable on datahub, should we choose to replicate your result. This includes code for: 

- code for data acquisition/scraping
- code for preprocessing
- training code (if appropriate)
- generative methods

Link each of these items to your .ipynb or .py files within this seection, and provide a brief explanation of what the code does. Reading this section we should have a sense of how to run your code.

### Code for Data Acquistion/Scraping
- code: https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group3/blob/master/code/DSC%20160%20Final%20Project%20-%20Lyric%20Scrape.ipynb
Our initial steps for acquiring our dataset first involved collecting the title of the songs within the URL format (i.e “Crazy in Love” would appear as “crazyinlove.html”) for the azlyrics.com website through its subsequent HTML page using the BeautifulSoup package. For our project, we encountered over 200 initial songs before we were able to narrow down to 127 songs. We manually cutoff the number of songs by the order they appear on azlyrics.com. It was then we added a line of code that extracted only unique values, meaning songs that appear twice were only counted as one song. 

With our list of songs ready, we then constructed a series of for-loops that would iterate through our list of songs. In order, our code would access the html page for the specific song, access the location of the actual song lyrics through div tags, and then subsequently scrape the lyrics located within another div tag. The BeautifulSoup package was again utilized throughout these steps. As the for-loops are running, the lyric information is being stored in both a dictionary and list. The completed list was then used to generate a text file storing all the lyrics as one string, while the dictionary was used to store the song title and its lyrics in a dataframe. It is from our experience using azlyrics.com that it is best not to scrape all 127 songs within one cell. Early scrapping attempts resulted in a temporary IP ban, thus we ran our final scraping algorithm in 6 groups or chunks of songs, to avoid any issues with the site.

### Code for Preprocessing and Feature Extraction
- code for lyric analysis and sentiment analysis: https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group3/blob/master/code/Lyric_Sentiment_Analysis.ipynb
Our first feature notebook looks at common words and sentiments in lyrics. This notebook looks at the length of lyrics (in characters and words), as well as what percentage of the lyrics are romantic, positive, and negative. Running all cells should conduct feature extraction and produce plots that explain the differences between the two sets of lyrics.

- code for ngram modeling: https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group3/blob/master/code/N_Gram%20Modeling.ipynb
This notebook focuses on extracting different features from the set of lyrics and songs we have. The notebook looks at most common phrases, data visualizations, TF-IDF vectors, as well as a probability model. Running all these cells should give a sense of the lyrics’ features as well as some statistics.


### Code for Model
- https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group3/blob/master/code/Beyonce%20-%20RNN%20Text%20Generation.ipynb
We used a Recurrent neural network in order to generate our Beyonce song lyrics.  We implemented one similar to the example on the Tensorflow website and the one given by the Professor. An RNN is a class of neural networks that is very powerful for modeling sequence data like natural language, which is what we used it for. An RNN works by maintaining an internal state that encodes information about the characters it has seen so far while using a prediction for loop which iterates over the sequence. One important parameter used for training our data was the number of epochs. An epoch is a measure of the number of times all of the training vectors are used once to update the weights. In our model, we used a different amount of epochs when training in order to see how our generated text improved and worsened. We started by using 10 epochs, which gave us text that did not have many complete words and were mostly random letters that did not make sense. Then we used 30 epochs, which gave us clear words, but was not necessarily coherent. We wanted to see if our generated text could get better than this so we decided to try using 50 epochs, but this gave us text that was directly pulled from her songs, with many phrases being actual lyrics. We realized that using 50 epochs made us overfit our model, and therefore we believed that 30 epochs would be the ideal number for our RNN. 

When given an input starting string (in our case, we used the popular lyric “Drunk in Love”), the model returns a bunch of generated text. The text generated is dependent upon the conditions we set for our model (number of epochs, number of characters, songs trained on, etc.), and as mentioned above, we were able to adjust these conditions to create the best model.



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

After running all of Beyonce’s lyrics through an RNN model, we generated a song that made legible words, but were not grammatically correct. Since the API only allowed us to input the amount of characters we wanted, we inputted the value of mean characters in a Beyonce song. We ran the model through multiple epoch values to increase the training time and we found that the higher epoch value did give us a better result, but it took an exponentially longer amount of time. Lower epoch values could not even create legible words so it made sense to increase the value and we settled it upon 30 Epochs because it generated real words without overfitting our data. We then quantitatively compared both the existing lyrics and the newly generated lyrics and found a decent amount of similarity between them. 

This is culturally innovative because Beyonce is one of the biggest musical artists of our generation, with over 100 million records sold worldwide. Moreover, music is a really important part of many people’s lives, which is why being able to generate music that resembles a particular artist is very interesting. As technology is improving, this type of generative art will only improve, making the generated song sound more and more realistic. Moreover, text generation technology gives musicians another way to approach their art, by giving them new ideas and inspiration for their work.

One thing that differentiates our “generated song” is that the computational approach bases these words on the given training set of past songs. We are basing our lyrics purely on past lyrics from Beyonce, so we won’t necessarily “create” new words to use. Generally, when artists are creating new songs, they don’t necessarily look at previous songs and copy the words/phrases from a previous song.  Most artists don’t reuse lyrics because each song represents a different period in life or message that they’re trying to present. Musical artists strive to be different from one another and in order to do so lyrically, they might try to follow their inspirations and feelings that surround them rather than looking at their own previous works. 

Our results relate to broader social, cultural, economic, and political issues as politically, because our lyrics are only trained on lyrics from preexisting songs, there is a limitation on how political the song can be and what social or political issues it can touch upon. If there is a new movement taking form for example (similar to what is going on right now), then the song won’t have the capacity or ability to address it. A lot of songs emerge from a specific social or political context and a lot of artists use music as a powerful outlet, so generative lyrics are limited and do not have this same creative potential. Also, with the emergence of generative lyrics, there will be larger implications in terms of the power and capacity of music, for example there are lots of protest songs right now but the extent to which these can emerge and have powerful effects will be lessened. Knowing that our results relate broadly to political issues also highlights the ethical concerns around generative art and music. Ethical concerns regarding our generative project would primarily revolve around ownership of the newly generated lyrics. With the current copyrights in the United States, there is no clear answer as to whether or not the artist used to train the algorithm or the algorithm itself would actually own the generated song. In our case, there currently is no specific law that would grant intellectual property rights of our generated results to Beyonce. However, another ethical issue surrounding generating lyrics and songs, would be the licensing on the songs used to train the algorithm. There is currently a debate about using copyrighted songs for training purposes with arguments suggesting that this could be a more clearly defined issue. But according to theverge.com, their article on AI in the music industry found experts to still be unsure about how exactly this could be a problem. At any rate, the lack of specificity in copyright laws when it comes to AI and other generative technology will surely raise more questions than answers for the time being.  

We could expand this work by using a word based RNN model instead of a character based one. This might allow us to gain more coherent lyrics or lyrics that better match Beyonce’s lyrical style. Additionally, we could add more metrics for comparison between the generated lyrics and the existing lyrics to better quantify the differences. This could include a more robust sentiment analysis, along with a deeper extraction of the themes and ideas that she sings about. Based on our peer feedback, we could also incorporate a GPT-2 model instead of, or in addition to, the results from our RNN model. Many of our peers thought that this would produce more coherent results, which would also help in comparing the newly generated lyrics to the existing lyrics.


## Team Roles

Provide an account of individual members and their efforts/contributions to the specific tasks you accomplished.

- Austin Le: Worked on feature extraction, Final Report
- Justin Kang: Worked on data source collection, Final Report
- Sravya Voleti: Worked on the RNN model, Final Report
- Sruthi Vedantham: Worked on feature extraction, Final Report
- Zahra Masood: Worked on the RNN model, Final Report


## Technical Notes and Dependencies

Any implementation details or notes we need to repeat your work. 
- Additional libraries you are using for this project
- Does this code require other pip packages, software, etc?
- Does this code need to run on some other (non-datahub) platform? (CoLab, etc.)

We primarily used datahub and the jupyter notebook attachment of the Anaconda, a popular data science software environment to complete our project. Additional resources utilized in this project included the Tensorflow package for our model creation and the Wordcloud package for visualization purposes. No further software or package installations were necessary beyond two mentioned packages, to complete the project and all other packages found in Anaconda should be adequate to repeat the steps in our project. In terms of the data collection, if the same scraping techniques are to be implemented we would advise scraping the lyrics in groups rather than in one instance. The process will take more time, but avoids the possibilities of an IP ban from the source site.  

## Reference

All references to papers, techniques, previous work, repositories you used should be collected at the bottom:
- Papers
- Repositories
- Blog posts

- Papers:
  - Exploring the unreasonable effectiveness of RNNs: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
  - Utilizing LSTMs and CNNs for creation of song lyrics: https://techxplore.com/news/2019-01-song-lyrics-style-specific-artists.html
  - Creating word-based RNNs using tensorflow: https://github.com/hunkim/word-rnn-tensorflow 
  
- Repositories: 
  - Professor Twomey’s RNN and generative text example: https://github.com/roberttwomey/dsc160-code/blob/master/examples/text-generation-rnn.ipynb

- Tutorials: 
  - TensorFlow: Recurrent Neural Networks (RNN) with Keras: https://www.tensorflow.org/guide/keras/rnn
  - TensorFlow: Text Generation with an RNN: https://www.tensorflow.org/tutorials/text/text_generation
  
- Blog posts:
  - TheVerge: Warnings about AI in the Music Industry: https://www.theverge.com/2019/4/17/18299563/ai-algorithm-music-law-copyright-human
