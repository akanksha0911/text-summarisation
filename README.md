# Explore Text Summarization 

[Project Writeup](https://github.com/akanksha0911/text-summarisation-webapp/blob/main/docs/Project_%20Speech_Text%20to%20Text%20Summarization.pdf)

[Project Presentation Slides](https://github.com/akanksha0911/text-summarisation-webapp/blob/main/docs/PPT-Text%20Summarization.pdf)


![Text-Summarization](https://github.com/akanksha0911/text-summarisation-webapp/blob/main/images/Text-Summarization.png)

Description: It intends to provide concise, accurate, easy to read, and comprehensive summaries of lengthy audio files or text files that one can read within a few minutes.

Reading a condensed summary of details from a long audio file/text file can give anyone a sense of what file focuses on or help catch up on hours’ worth of content in minutes.

In the case of online content available today, it is massive. Books, articles, blogs, and audios like podcasts, broadcasting of news on radio channels, speeches, etc. All the audio files data obtained from these sources are not effectively a valuable means of gathering information every time. It is always effective if the data is obtained from the condensed content i.e. focusing more on essential points than the entire content.
This application can be intensively used in summarizing business meetings. It serves in more efficient progress tracking in project meetings and facilitation of learning using online courses. As a voice-to-text application, people with hearing disabilities could benefit from summarization to keep up with content in a more productive way.

An audio or text file will be taken as input to progress with this application, and data summarization processes will be implemented to get condensed text output. 

![Arch](https://user-images.githubusercontent.com/77387431/143853833-f97cb2ed-f92d-4d8e-824d-4d28896f2bbf.png)


# Setup
This codebase is based on python3 (3.8.5). Other python versions should work seamlessly.

## Install Requirements
```shell
pip3 install -r requirements.txt
```

## Pip3 Freeze
In any circumstance, the requirements did not meet the setup then try the freeze based requirement
```shell
pip3 install -r pip3freeze.txt
```


## Get started

### Start
```shell
./run.sh articles/msft.txt
```

### To just generate the best summary
```shell
python3 best_summary.py articles/msft.txt
```

### Store reports
This is resource intensive and time consuming as each pegasus pretrained model is loaded and run.
So storing the created reports is advisable.

```shell
python3 best_summary.py articles/msft.txt > reports/msft.txt
```

### Look at reports
To see just the best summary only
```shell
tail -1 reports/msft.txt
```

# Contributors
akanksha0911 </br>
Karishma-Kuria
