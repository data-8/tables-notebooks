"use strict;"
let hub = "datahub.berkeley.edu";
let repo = "tables-notebooks";
let branch = "gh-pages";
function makeHubLink(notebook) {
    document.write(`<a href=http://${hub}/user-redirect/interact?repo=${repo}&branch=${branch}&path=${notebook}>${notebook}</a>`);
}

