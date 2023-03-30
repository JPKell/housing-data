function clearCache () {
  fetch('/clear-cache').then(console.log("Cache cleared"))
}