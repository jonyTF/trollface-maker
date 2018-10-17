const express = require('express');
const app = express();
const http = require('http').Server(app);
//const PythonShell = require('python-shell');
const spawn = require('child_process').spawn;
const py = spawn('python', ['make_trollface.py'])
const multer = require('multer');
const upload = multer({ dest: 'img/' });
const FileCleaner = require('cron-file-cleaner').FileCleaner;
//const ejs = require('ejs');
const waitUntil = require('wait-until');
const fs = require('fs');

var processingFile = false;

app.use(function(req, res, next) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});  

app.use(express.static(__dirname + '/www'));
app.use('/img', express.static(__dirname + '/img'));
app.get('/', function(req, res) {
    res.sendFile(__dirname + '/www/index.html');
});

py.stdout.on('data', function(data) {
    console.log(data.toString());
});

py.stderr.on('data', function(data) {
    console.log(data.toString());
});

app.post('/make_trollface', upload.single('pic'), function(req, res, next) {
    processingFile = true;
    var file = req.file;
    var newpath = file.path+'.jpg';
    console.log(file);

    py.stdin.write(JSON.stringify([file.path, newpath]) + '\n');
    //py.stdin.end();

    res.write(JSON.stringify({'filename': file.filename + '.jpg'}));
    res.end();

    /*const options = {
        mode: 'json',
        args: [file.path, newpath]
    };
    console.log('Running make_trollface.py with options:', options);
    PythonShell.run('make_trollface.py', options, function(err, results) {
        if (err) {
            throw err;
        }
        console.log('Trollface_count: ', results[0].trollface_count)
        console.log('Done processing file!')
        //res.contentType('image/jpeg');
        res.write(JSON.stringify({'filename': file.filename + '.jpg'}));
        res.end();

        processingFile = false;*/
        /*fs.unlink(newpath, function(err) {
            if (err) throw err;
            console.log(newpath, 'was deleted');
        });*/
    //});
});

app.post('/is_processing_trollface', function(req, res) {
    console.log('isprocessing?!?');
    res.write(JSON.stringify({'processingFile': processingFile}));
    res.end();
});

app.get('/load', function(req, res) {
    //res.sendFile(__dirname + '/www/load.html');
    //res.end();
    
    const filename = req.query.img;

    waitUntil()
        .interval(500)
        .times(120)
        .condition(function(cb) {
            process.nextTick(function() {
                cb(fs.existsSync(__dirname + '/img/' + filename) ? true : false);
            })
        })
        .done(function(result) {
            res.redirect('/img/' + filename);
        });
});

// Delete all files older than an hour
var fileWatcher = new FileCleaner(__dirname+'/img/', 3600000, '* */15 * * * *', {
    start: true,
    blacklist: '/\.init/'
});

http.listen(process.env.PORT || 1266, function() {
    var port = http.address().port;
    console.log('Server listening on port ', port)
});

 