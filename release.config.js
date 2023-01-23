// https://semantic-release.gitbook.io/semantic-release/support/faq#can-i-use-semantic-release-to-publish-non-javascript-packages
module.exports = {
  "branches": ["main"],
  "plugins": [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    [
      "@semantic-release/changelog",
      {
        "changelogFile": "CHANGELOG.md"
      }
    ],
    [
      "@semantic-release/git",
      {
        "assets": ["CHANGELOG.md"]
      }
    ],   
    "@semantic-release/github",
    ["@semantic-release-plus/docker", {
      "name": "gadicc/diffusers-api"      
      }]
  ]
}