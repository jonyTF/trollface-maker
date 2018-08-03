const express = require('express')
const app = express()
const http = require('http').Server(app)
const bodyParser = require('body-parser')
const PythonShell = require('python-shell');

http.listen(process.env.PORT || 1266, function() {
    console.log('Server listening on port 1266')
});

/*const options = {
    mode: 'json',
    args: ['img/mich.png']
};

PythonShell.run('main.py', options, function(err, results) {
    if (err) throw err;
    console.log(results[0].img_path);
});*/

 