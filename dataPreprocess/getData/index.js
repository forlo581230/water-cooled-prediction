var util = require('util');
var request = require('request');
var moment = require('moment');
var fs = require('fs');
var QueryString = require('querystring');

// http://203.75.178.71:8087/login

function getUnixTime(momentDate) {
    var unixTime = momentDate.unix();
    // var time = moment.unix(1555257600000/1000)
    // console.log(momentDate);
    return (unixTime * 1000).toString() + 'ms';
}

function getData(queryString) {
    return new Promise((resolve, reject) => {
        console.log('http://203.75.178.240:8086/query?' + queryString);
        request({
            url: ('http://203.75.178.240:8086/query?' + queryString),
            json: true
        }, (err, res, body) => {
            if (typeof body.results[0].series !== 'undefined') {
                for (let i = 0; i < body.results[0].series[0].values.length; i++) {
                    body.results[0].series[0].values[i][0] = moment(body.results[0].series[0].values[i][0]).unix();
                    // console.log(body.results[0].series[0].values[i][0]);
                }
                resolve(JSON.stringify(body.results[0].series[0]));
            } else {
                resolve([]);
            }
            // console.log(body.results[0].series[0].columns)
            // console.log(body.results[0].series[0].values)
        })
    })

}

async function main(argv) {

    const username = argv[2]
    const password = argv[3]
    if (typeof username === "undefined" || typeof password === "undefined") {
        console.log('please input username and password.')
        return;
    }
    // bank, "013137800024771"
    const address = ["013157800087578"]
    // world gym, xin-dian-zhong-hua-yin-shua
    // const address = ["013137800024763","013157800087602"]
    // lin-xxx
    // const address = ["013137800024771"]

    let dir = "../datasets/rawData/"

    for (let mac of address) {
        if (!fs.existsSync(dir + mac)) {
            fs.mkdirSync(dir + mac);
        }

        let selectString = '';
        let queryString = ``;
        let fileName = '';
        let currentDate = moment();
        let minDate = moment('2019-01-10'); //10-11
        let maxDate = moment('2019-01-11');

        while (currentDate.isAfter(minDate)) {

            selectString =
                `SELECT "冰水入口溫度", "冰水出口溫度", "冷卻水入口溫度", "冷卻水出口溫度", "右機LEV開度Step", "右機低壓", "右機吐出溫度", "右機電流", "右機高壓", "右耗電", "右飽和冷凝溫度", "右飽和蒸發溫度", "壓縮機頻率", "左機LEV開度Step", "左機低壓", "左機吐出溫度", "左機電流", "左機高壓", "左耗電", "左飽和冷凝溫度", "左飽和蒸發溫度", "設溫" FROM "PQ" WHERE `;
            // selectString = `SELECT * FROM "PQ" WHERE `;
            selectString += util.format(`("address"= '%s')`, mac);
            selectString += util.format(` AND time >= %s AND time < %s`, getUnixTime(minDate), getUnixTime(maxDate));

            // selectString +=
            // ' AND time >= '+getUnixTime(minDate)+' AND time < '+getUnixTime(maxDate);

            let obj = {
                "u": username,
                "p": password,
                "db": "nbiotdb",
                "q": selectString,
            };
            queryString = QueryString.stringify(obj);


            fileName = minDate.format('YYYY-MM-DD') + '.json';
            datas = await getData(queryString);

            if (datas.length > 0) {
                fs.writeFile('../datasets/rawData/' + mac + '/' + fileName, datas, function (err) {
                    if (err)
                        console.log(err);
                    else
                        console.log(fileName, 'write operation complete.');
                });
            }

            minDate.add(1, 'days');
            maxDate.add(1, 'days');
        }

    }




}

main(process.argv);