 YouTube Video Summarization with BART
 
Overview:

In this video, we'll explore the fascinating world of YouTube video summarization using BART (Bidirectional and Auto-Regressive Transformers), a state-of-the-art natural language processing model. As the vastness of online content continues to grow, the need for efficient summarization techniques becomes increasingly essential. BART offers a powerful solution by intelligently condensing lengthy videos into concise summaries, enabling users to grasp key insights and information in a fraction of the time. Join us as we delve into the process of leveraging BART for YouTube video summarization, from data collection to model implementation, and discover how this technology is revolutionizing the way we consume and engage with online video content.

Key Features of YouTube Video Summarization Using BART:

1. BART Model Integration:
   - Utilizes the Bidirectional and Auto-Regressive Transformers (BART) model, known for its effectiveness in natural language processing tasks.
   - BART excels in summarization tasks due to its ability to generate coherent and informative summaries.

2. Abstractive Summarization:
   - Implements abstractive summarization techniques to generate concise yet informative summaries of YouTube videos.
   - Goes beyond simple extraction of keywords or sentences, producing summaries that capture the essence of the video content.

3. Multi-Modal Inputs:
   - Accommodates multi-modal inputs, including audio, video, and subtitles, to create comprehensive summaries.
   - Integrates information from various modalities to ensure that the generated summaries are rich and contextually relevant.

4. Fine-Tuning for YouTube Content:
   - Fine-tunes the BART model on a dataset of YouTube videos to enhance its performance specifically for this domain.
   - Adapts the model to recognize and summarize the unique characteristics of YouTube content, such as diverse topics and styles.

5. User-Driven Customization:
   - Provides options for users to customize the summarization process based on preferences and requirements.
   - Allows users to specify desired length, level of detail, and inclusion/exclusion of specific content elements in the summaries.

6. Real-Time Summarization:
   - Enables real-time summarization of YouTube videos, allowing users to quickly access concise summaries without watching the entire video.
   - Enhances user experience by providing efficient access to relevant information and saving time.

7. Integration with YouTube API:
   - Seamlessly integrates with the YouTube API to retrieve video content and metadata for summarization.
   - Ensures compatibility and ease of use for users familiar with the YouTube platform.

8. Evaluation Metrics and Feedback Mechanism:
   - Incorporates evaluation metrics to assess the quality and coherence of generated summaries.
   - Includes a feedback mechanism for users to provide input on the accuracy and relevance of the summaries, facilitating continuous improvement.

9. Scalability and Accessibility:
   - Designed to be scalable, allowing for summarization of large volumes of YouTube videos efficiently.
   - Ensures accessibility across different devices and platforms, making it widely available to users worldwide.
  
  CODE
  {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "# Install required packages\n",
        "!pip install youtube_transcript_api\n",
        "!pip install transformers\n",
        "\n",
        "# Import necessary libraries\n",
        "import youtube_transcript_api\n",
        "from youtube_transcript_api import YouTubeTranscriptApi\n",
        "import nltk\n",
        "import re\n",
        "from nltk.corpus import stopwords\n",
        "import sklearn\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "import transformers\n",
        "from transformers import BartTokenizer, BartForConditionalGeneration"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PTntK4tejYw6",
        "outputId": "8a46db14-4989-4cd7-cff4-4d622815d33e"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: youtube_transcript_api in /usr/local/lib/python3.10/dist-packages (0.6.2)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from youtube_transcript_api) (2.31.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->youtube_transcript_api) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->youtube_transcript_api) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->youtube_transcript_api) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->youtube_transcript_api) (2024.2.2)\n",
            "Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.38.2)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.13.3)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.19.3 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.25.2)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.0)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.1)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2023.12.25)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.31.0)\n",
            "Requirement already satisfied: tokenizers<0.19,>=0.14 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.15.2)\n",
            "Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.2)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.2)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers) (2023.6.0)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers) (4.10.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.2.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Collect user input for the YouTube link\n",
        "link = input(\"Enter the link here : \")\n",
        "unique_id = link.split(\"=\")[-1]  # Extract the video ID from the link\n",
        "\n",
        "# Retrieve the transcript for the YouTube video\n",
        "sub = YouTubeTranscriptApi.get_transcript(unique_id)\n",
        "subtitle = \" \".join([x['text'] for x in sub])  # Combine transcript into a single string\n",
        "\n",
        "# Initialize the BART tokenizer and model\n",
        "tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')\n",
        "model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')\n",
        "\n",
        "# Tokenize the input subtitle and generate a summary\n",
        "input_tensor = tokenizer.encode(subtitle, return_tensors=\"pt\", max_length=512)  # Tokenize the subtitle\n",
        "outputs_tensor = model.generate(input_tensor, max_length=160, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)  # Generate summary\n",
        "\n",
        "# Decode the generated summary and print it\n",
        "print(\"SUMMARY:\")\n",
        "print(tokenizer.decode(outputs_tensor[0]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FwXhYLkLkqDQ",
        "outputId": "27d08e91-1255-46ee-8596-6cefbe800143"
      },
      "execution_count": 18,
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Enter the link here : https://www.youtube.com/watch?v=et7XvBenEo8\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "SUMMARY:\n",
            "</s><s>The three body problem is famous for being difficult to solve. But actually it's been solved many times, and in ingenious ways. Some of those solutions are incredibly useful, and some are incredibly bizarre. Physics - and arguably all of science changedforever in 1687 when Isaac Newton published his Principia. Within it were equations ofmotion and gravity that transformed our erratic-seeming cosmos into a perfectly tuned machine of clockwork predictability. Despite the beauty of Newton’s equations, they lead to a simple solution for planetary motion in only one case - when two and only two bodies orbit each other sans any other gravitational influence in the universe.</s>\n"
          ]
        }
      ]
    }
  ]
}

