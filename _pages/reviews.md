<!-- ---
layout: page
title: reviews
permalink: /reviews/
description:
nav: true
nav_order: 2
---

# Instructions to review the blogposts

1. **Git fetch the upstream branches**:

* If you haven't forked the [staging](https://github.com/gram-blogposts/staging) repository yet, then you should do it.
* Once the repository is forked, you need to fetch the branch called `reviews`. For this, the upstream repository should be added as remote.

   ```bash
   $ git remote add upstream https://github.com/gram-blogposts/staging.git
   $ git fetch upstream
   $ git checkout -b reviews upstream/reviews
   ```
* You are now in the `reviews` branch.

2. **Compile the site locally**:

* The next step is to compile locally the website. You may want to update `bundle`. Compiling all the blog posts will take some time.

   ```bash
   $ bundle update
   $ bundle exec jekyll serve
   ```

* After a few lines of warnings, you should have the server address that you can enter in your favorite browser to check the website.

    ```bash
    Auto-regeneration: enabled 
    Server address: http://127.0.0.1:4000
    Server running... press ctrl-c to stop.
    ```

3. **Give constructive feedback on our Google form**:

* Click on the index `reviews` of the website that is built locally on your computer.
* You can click on the blog posts you would like to review.
* Your own blog post is also visible -- if you notice that it's not rendering the way you want, don't worry. You can still modify your pull request to add, for examples, the missing references, or fix the figures. 
* We ask the reviewers to judge the clarity and the pedagogical aspects of the blog post, by completing the following [Google Form.](https://forms.gle/Y69471ZL98JYY59LA)

Thank you for your help! -->