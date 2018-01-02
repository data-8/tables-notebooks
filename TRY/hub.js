"use strict;"
let hub = "mybinder.org";
let repo = "data-8/tables-notebooks";
let branch = "gh-pages";

function makeHubLink(notebook) {
    document.write(`<a href=https://${hub}/v2/gh/${repo}/${branch}?filepath=${notebook}>${notebook}</a>`);		   
}




