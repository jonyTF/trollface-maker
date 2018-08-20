const express = require('express');
const app = express();
const http = require('http').Server(app);
const PythonShell = require('python-shell');
const multer = require('multer');
const upload = multer({ dest: 'img/' });
const FileCleaner = require('cron-file-cleaner').FileCleaner;

app.use(function(req, res, next) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});  

app.use(express.static(__dirname + '/www'));
app.get('/', function(req, res) {
    res.sendFile(__dirname + '/www/index.html');
});

app.post('/make_trollface', upload.single('pic'), function(req, res, next) {
    var file = req.file;
    var newpath = file.path+'.jpg';
    console.log(file);
    const options = {
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
        /*fs.unlink(newpath, function(err) {
            if (err) throw err;
            console.log(newpath, 'was deleted');
        });*/
    });
});

app.use('/img', express.static(__dirname+'/img'));

// Delete all files older than an hour
var fileWatcher = new FileCleaner(__dirname+'/img/', 3600000, '* */15 * * * *', {
    start: true,
    blacklist: '/\.init/'
});

http.listen(process.env.PORT || 1266, function() {
    var port = http.address().port;
    console.log('Server listening on port ', port)
});

 