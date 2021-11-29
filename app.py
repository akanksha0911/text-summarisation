from __future__ import print_function

import json

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

import abstractive_summaries
import best_summary
import extractive_summary
import rouge_scorer
import time
import boto3

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


@app.route('/transcribe', methods=['POST'])
def transcribe_file():
    req_json = request.json

    path = req_json['path']
    transcribe = boto3.client('transcribe')
    job_name = str(time.time())
    job_uri = path
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        LanguageCode='en-US'
    )
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Not ready yet...")
        time.sleep(5)
    print(status)

    r = requests.get(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
    response = json.loads(r.text)
    print(response)
    print(response['results']['transcripts'][0]['transcript'])
    return jsonify({'transcript': response['results']['transcripts'][0]['transcript']})


@app.route('/extractive', methods=['POST'])
def post_extractive():
    req_json = request.json

    text = req_json['text']
    summary = extractive_summary.generate_summary(text)
    return summary


@app.route('/rogueScore', methods=['POST'])
def rogue_score():
    req_json = request.json

    summary_a = req_json['summary_a']
    summary_b = req_json['summary_b']

    score = best_summary.score_summary(summary_a, summary_b)
    return jsonify(score)


@app.route('/abstractive', methods=['POST'])
def post_abstractive():
    print(request.json)
    req_json = request.json

    text = req_json['text']
    model = req_json['model']

    if model is 'arawat/pegasus-custom-xsum':
        return abstractive_summaries.generate_summary_custom_model(text,model)
    return abstractive_summaries.generate_summary(text, model)


@app.route('/bestSummary', methods=['POST'])
def post_best_summary():
    req_json = request.json

    text = req_json['text']
    return best_summary.best_summary(text)


if __name__ == '__main__':
    app.run()
