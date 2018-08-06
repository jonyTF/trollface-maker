const express = require('express');
const app = express();
const http = require('http').Server(app);
const bodyParser = require('body-parser');
const formidable = require('formidable');
const fs = require('fs');
const PythonShell = require('python-shell');

function guid() {
    function s4() {
        return Math.floor((1 + Math.random()) * 0x10000)
            .toString(16)
            .substring(1);
    }
    return s4() + s4() + s4() + s4() + s4() + s4() + s4() + s4();
}

function getFileExtension(filename) {
    return filename.split('.').pop();
}

app.get('/', function(req, res) {
    res.sendFile(__dirname + '/www/index.html');
});

app.post('/make_trollface', function(req, res) {
    var form = new formidable.IncomingForm({uploadDir: __dirname + '/img'});
    form.parse(req, function(err, fields, files) {
        var oldpath = files.filetoupload.path;
        var newpath = __dirname + '/img/' + guid() + '.' + getFileExtension(files.filetoupload.name)
        fs.rename(oldpath, newpath, function(err) {
            if (err) {
                throw err;
            }
            console.log('File submitted!\nProcessing file...');
            //res.write('File submitted!');
            //res.write('Processing file...');
            
            const options = {
                mode: 'json',
                args: [newpath, newpath]
            };
            console.log('Running main.py with options:', options);
            PythonShell.run('main.py', options, function(err, results) {
                if (err) {
                    throw err;
                }
                console.log('Trollface_count: ', results[0].trollface_count)
                console.log('Done processing file!')
                //res.write('Done processing file!');
                res.sendFile(newpath);
                res.end();
                /*fs.unlink(newpath, function(err) {
                    if (err) throw err;
                    console.log(newpath, 'was deleted');
                });*/
            });
        });
    });
});

http.listen(process.env.PORT || 1266, function() {
    var port = http.address().port;
    console.log('Server listening on port ', port)
});
/*const options = {
    mode: 'json',
    args: ['img/mich.png']
};

PythonShell.run('main.py', options, function(err, results) {
    if (err) throw err;
    console.log(results[0].img_path);
});*/

 