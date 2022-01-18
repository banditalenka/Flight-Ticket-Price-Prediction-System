function onClickedInputDetails() {
    console.log("Input Details clicked");
      var book = document.getElementById("Book_Time").value;
      if(book==""){
        alert(" Enter Booking Date time")
        return
    }
      var dep = document.getElementById("Dep_Time").value;
      if(dep==""){
        alert(" Enter Departure Date time")
        return
    }
    /* DEP TIME Check*/
      if(dep>book){
          console.log("CORRECT INPUT");
      }
      else{
        console.log("WRONG INPUT");
        alert("Departure Date Cannot Be Before Booking Date");
        return;
      }
    /* DEP TIME Check*/
      var arr = document.getElementById("Arr_Time").value;
      if(arr==""){
        alert(" Enter Arrival Date time")
        return
    }
    /* ARR TIME Check*/
    if(dep<arr){
        console.log("CORRECT INPUT");
    }
    else{
      console.log("WRONG INPUT");
      alert("Arrival Date Cannot Be Before Departure Date Time");
      return;
    }
    /* ARR TIME Check*/
      var flight = document.getElementById("flight").value;
      var source = document.getElementById("source").value;
      var destination = document.getElementById("destination").value;
      /* Source Destination Check*/
      if(source == "Delhi" && destination == "New Delhi"){
        console.log("WRONG INPUT");
        document.getElementById("result").innerHTML="";
        alert("Source And Destination Cannot Be Same");
        return;
    }
    else if(source!=destination){
        console.log("CORRECT INPUT");
    }
    else{
      console.log("WRONG INPUT");
      document.getElementById("result").innerHTML="";
      alert("Source And Destination Cannot Be Same");
      return;
    }
    /* Source Destination Check*/
    /* Stoppage value*/
      var stops = document.getElementsByName('stoppage');
      var stoppage=null;
        for(i = 0; i < stops.length; i++) {
            if(stops[i].checked){
                stoppage=stops[i].value;
            }
        }
    /* Stoppage value*/
    /* Stoppage value Check*/
    if(stoppage == null){
        document.getElementById("result").innerHTML="";
        alert("Choose Number of Stops");
        return;
      }
    /* Stoppage value Check*/
      console.log("Booking date : " + book)
      console.log("Departure Date : " + dep)
      console.log("Arrtival Date : " + arr)
      console.log("Flight : " + flight)
      console.log("Source : " + source)
      console.log("Destination :" + destination)
      
      console.log("Stoppage :" + stoppage)
      document.getElementById("result").innerHTML="Booking date : " + book + "<br>" 
      +"Departure Date : " + dep+ "<br>" +"Arrival Date : " + arr+"<br>"+ "Flight : " + flight 
      + "<br>"+"Source : " + source + "<br>" + "Destination :" + destination + "<br>"+ "Stoppage :" 
      + stoppage + "<br>" + "Estimation : Under Progress";
    }
    
    
    function onPageLoad() {
        console.log( "document loaded" );
        var url = "http://127.0.0.1:5000/get_flight_names";
        // Use this if you are NOT using nginx which is first 7 tutorials
        //var url = "/api/get_location_names"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
        $.get(url,function(data, status) {
            console.log("got response for get_flight_names request");
            if(data) {
                var flights = data.flights;
                var flight = document.getElementById("flight");
                $('#flight').empty();
                for(var i in flights) {
                    var opt = new Option(flights[i]);
                    $('#flight').append(opt);
                }
            }
        });
        url = "http://127.0.0.1:5000/get_source_names";
        $.get(url,function(data, status) {
            console.log("got response for get_source_names request");
            if(data) {
                var sources = data.sources;
                var source = document.getElementById("source");
                $('#source').empty();
                for(var i in sources) {
                    var opt = new Option(sources[i]);
                    $('#source').append(opt);
                }
            }
        });
        url = "http://127.0.0.1:5000/get_destination_names";
        $.get(url,function(data, status) {
            console.log("got response for get_destination_names request");
            if(data) {
                var destinations = data.destinations;
                var destination = document.getElementById("destination");
                $('#destination').empty();
                for(var i in destinations) {
                    var opt = new Option(destinations[i]);
                    $('#destination').append(opt);
                }
            }
        });
    
    /*    n =  new Date();
    y = n.getFullYear();
    m = n.getMonth() + 1;
    d = n.getDate();
    hh=n.getHours();
    mm=n.getMinutes();
    mindate=y + "-" + m + "-" + d + "T"+hh+":"+mm;
    document.getElementById("date").innerHTML = y + "-" + m + "-" + d + "T"+hh+":"+mm;*/
    }
    
    function onClickedEstimatePrice() {
        console.log("Estimate price button clicked");
        var book = document.getElementById("Book_Time").value;
        if(book==""){
            alert(" Enter Booking Date time")
            return
        }

        var dep = document.getElementById("Dep_Time").value;
        if(dep==""){
            alert(" Enter Departure Date time")
            return
        }
        if(dep>book){
            console.log("CORRECT INPUT");
        }
        else{
          console.log("WRONG INPUT");
          alert("Departure Date Cannot Be Before Booking Date");
          return;
        }
        var arr = document.getElementById("Arr_Time").value;
        if(arr==""){
            alert(" Enter Arrival Date time")
            return
        }
        if(dep<arr){
            console.log("CORRECT INPUT");
        }
        else{
          console.log("WRONG INPUT");
          alert("Arrival Date Cannot Be Before Departure Date Time");
          return;
        }
        var flight = document.getElementById("flight").value;
        var source = document.getElementById("source").value;
        var destination = document.getElementById("destination").value;

        if(source == "Delhi" && destination == "New Delhi"){
            console.log("WRONG INPUT");
            document.getElementById("result").innerHTML="";
            alert("Source And Destination Cannot Be Same");
            return;
        }
        else if(source!=destination){
            console.log("CORRECT INPUT");
        }
        else{
          console.log("WRONG INPUT");
          document.getElementById("result").innerHTML="";
          alert("Source And Destination Cannot Be Same");
          return;
        }

    /* Stoppage value*/
    var stops = document.getElementsByName('stoppage');
    var stoppage=null;
      for(i = 0; i < stops.length; i++) {
          if(stops[i].checked){
              stoppage=stops[i].value;
          }
      }
  /* Stoppage value*/
  if(stoppage == null){
    document.getElementById("result").innerHTML="";
    alert("Choose Number of Stops");
    return;
  }
      
        var url = "http://127.0.0.1:5000/predictFightPrice"; //Use this if you are NOT using nginx which is first 7 tutorials
        //var url = "/api/predict_home_price"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
      
        $.post(url, {
            Dep_Time : dep,
            Arrival_Time : arr,
            stops : stoppage,
            airline : flight,
            Source : source,
            Destination : destination,
            Book_Time : book
        },function(data, status) {
            console.log(data.estimated_price);
            document.getElementById("result").innerHTML="Estimated Price : "+ data.estimated_price.toString();
            console.log(status);
        });
      }
    
      
      $(document).ready(function(){
        elem1 = document.getElementById("Book_Time")
        elem2 = document.getElementById("Dep_Time")
        elem3 = document.getElementById("Arr_Time")
        var iso = new Date().toISOString();
        var d = new Date();
        var input = d.toTimeString()
        var out = input.substring(0,input.length-34); 
        var minDate = iso.substring(0,iso.length-8);
        var checkdate = minDate;
        var checkdate = checkdate.substring(0,checkdate.length-5) + out;
        elem1.min = checkdate
        elem2.min = checkdate
        elem3.min = checkdate
    });

window.onload = onPageLoad;


    
  /*  window.onload = onPageLoad;*/