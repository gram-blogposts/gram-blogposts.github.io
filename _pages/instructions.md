---
layout: page
title: instructions
permalink: /instructions/
description:
nav: true
nav_order: 2
---

### Contents

- [Instructions for Submitting a Blog Post](#instructions-for-submitting-a-blog-post)
- [Review and Publication](#review-and-publication)
- [Further questions](#further-questions)

### Instructions for Submitting a Blog Post

1. **Fork and Rename**:

   - Navigate to the [staging repository](https://github.com/gram-blogposts/staging/).
   - Click on the "Fork" button of the repository page to create a copy of the repository under your GitHub account.
   - After forking, rename your repository to reflect the topic of your submission.


2. **Setup and Deployment**:

   - Clone your forked repository to your local machine.
   - Setup locally the webiste.
     Assuming you have [Ruby](https://www.ruby-lang.org/en/downloads/) and [Bundler](https://bundler.io/) installed on your system, try:

   ```bash
   $ git clone git@github.com:<your-username>/<your-repo-name>.git
   $ cd <your-repo-name>
   $ bundle install
   $ bundle exec jekyll serve
   ```

   - Your can locally access and modify the website on the local address: `http://127.0.0.1:4000/`.


3. **Preparing Your Submission**:
   You can copy and modify the [Distill Template]({% post_url 2018-12-22-distill %}), found under the `_posts` directory. Leave the original template as it is.

   - Create a Markdown or HTML file in the `_posts/` directory with the format `_posts/YYYY-MM-DD-your-submission.md`.
   - Add any static image to `assets/img/YYYY-MM-DD-your-submission/`.
   - Add any interactive plotly figures to `assets/plotly/YYYY-MM-DD-your-submission/`.
   - Put your citations into a bibtex file in `assets/bibliography/YYYY-MM-DD-your-submission.bib`.


4. **Submit a Pull Request**:

   - Before submitting, double-check that all personal identifiers have been removed from your blog post content.
   - Push your changes to your forked repository.
   - Go to your forked repository on GitHub and click on "New Pull Request".
   - Ensure the pull request is directed to the staging repository and name the pull request after your submission topic.
   - Provide a clear and concise description of your changes in the pull request description.


<!-- 5. **Fill out the Google Form**:
   - After submitting a pull request, we will gather all the submission's information through this Google Form:https://forms.gle/pMmPvAqCvVTQP6eo6
   - You will need to review anonymously two other blogposts or tutorial that will be assigned to you through another form that will be sent by email. You will have a week to submit your reviews, and the decision on your blogpost or tutorial will be made before the 8th of July. -->

### Review and Publication

- **Review**: Reviewers will assess only the live content of the blog. They are expected to avoid probing into the repository's historical data to maintain anonymity.
- **Camera-Ready Submission**: Upon acceptance, add your identifying information to the post. Submit a final pull request to the main repository for integration into the official blog. Track chairs may request additional edits before final approval.

### Further questions

- Our Blogpost track is directly inspired by the (amazing) [ICLR Blogpost track](https://iclr-blogposts.github.io/). If you have problems building your website or modifying any file, you can have a look at their [instructions](https://iclr-blogposts.github.io/2024/submitting/).
- Feel free to reach out to the organizers at:
   - organizers &#91;&#97;&#116;&#93; gram-workshop &#91;&#100;&#111;&#116;&#93; org
   - alison.pouplin &#91;&#97;&#116;&#93; aalto &#91;&#100;&#111;&#116;&#93; fi
