var map;
var markers = [];
var qtWidget;
var paths = [];
var coordInfoWindow = new google.maps.InfoWindow();
var Mtext;

function initialize() {

    var myOptions = {
        center: {lat: -34.397, lng: 150.644},
        streetViewControl: false,
        mapTypeId: google.maps.MapTypeId.ROADMAP,
        zoom: 8
    };

    map = new google.maps.Map(document.getElementById('map_canvas'),
        myOptions);

    new QWebChannel(qt.webChannelTransport, function (channel) {
        qtWidget = channel.objects.qGoogleMap;
    });

    google.maps.event.addListener(map, 'dragend', function () {
        var center = map.getCenter();
        qtWidget.mapIsMoved(center.lat(), center.lng());
    });
    google.maps.event.addListener(map, 'click', function (ev) {
        qtWidget.mapIsClicked(ev.latLng.lat(), ev.latLng.lng());
    });
    google.maps.event.addListener(map, 'rightclick', function (ev) {
        qtWidget.mapIsRightClicked(ev.latLng.lat(), ev.latLng.lng());
    });
    google.maps.event.addListener(map, 'dblclick', function (ev) {
        qtWidget.mapIsDoubleClicked(ev.latLng.lat(), ev.latLng.lng());
    });
}

map.data.setStyle(function (feature) {
    var strokeColor = feature.getProperty('color');
    var dist = feature.getProperty('distance');
    dist = parseFloat(dist).toFixed(2);
    coordInfoWindow.setContent(Mtext + "<br>Distância: " + dist + "m");
    feature.getGeometry().forEachLatLng(function (latlng) {
        coordInfoWindow.setPosition(latlng);
    });
    coordInfoWindow.open(map);
    return {
        strokeColor: strokeColor,
        strokeWeight: 3
    };
});

function gmap_addPath(filepath, text) {
    paths.push(map.data.addGeoJson(filepath));
    Mtext = text;
}

function gmap_clearPaths() {
    for (var i = 0; i < paths.length; i++) {
        features = paths[i];
        for (var j = 0; j < features.length; j++)
            map.data.remove(features[j]);
    }
    paths = [];
}

function gmap_setCenter(lat, lng) {
    map.setCenter(new google.maps.LatLng(lat, lng));
}

function gmap_getCenter() {
    return [map.getCenter().lat(), map.getCenter().lng()];
}

function gmap_setZoom(zoom) {
    map.setZoom(zoom);
}

function gmap_addMarker(key, latitude, longitude, parameters) {

    if (key in markers) {
        gmap_deleteMarker(key);
    }
    var coords = new google.maps.LatLng(latitude, longitude);
    parameters['map'] = map
    parameters['position'] = coords;
    var marker = new google.maps.Marker(parameters);
    google.maps.event.addListener(marker, 'dragend', function () {
        qtWidget.markerIsMoved(key, marker.position.lat(), marker.position.lng())
    });
    google.maps.event.addListener(marker, 'click', function () {
        qtWidget.markerIsClicked(key, marker.position.lat(), marker.position.lng())
    });
    google.maps.event.addListener(marker, 'dblclick', function () {
        qtWidget.markerIsDoubleClicked(key, marker.position.lat(), marker.position.lng())
    });
    google.maps.event.addListener(marker, 'rightclick', function () {
        qtWidget.markerIsRightClicked(key, marker.position.lat(), marker.position.lng())
    });
    markers[key] = marker;
    return key;
}

function gmap_moveMarker(key, latitude, longitude) {
    var coords = new google.maps.LatLng(latitude, longitude);
    markers[key].setPosition(coords);
}

function gmap_deleteMarker(key) {
    markers[key].setMap(null);
    delete markers[key]
}

function gmap_changeMarker(key, extras) {
    if (!(key in markers)) {
        return
    }
    markers[key].setOptions(extras);
}


function saveImage(filename) {
    html2canvas(document.querySelector('#map_canvas')).then(function (canvas) {
        console.log("Saving canvas: " + canvas);
        saveAs(canvas.toDataURL(), filename);
    });
    return filename
}

function saveAs(uri, filename) {
    console.log("Running fakery to download this")
    var link = document.createElement('a');
    if (typeof link.download === 'string') {
        link.href = uri;
        link.download = filename;
        //Firefox requires the link to be in the body
        document.body.appendChild(link);
        //simulate click
        link.click();
        //remove the link when done
        document.body.removeChild(link);
    } else {
        window.open(uri);
    }
}