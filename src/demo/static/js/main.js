function detect() {
    var code = document.getElementById("code").value;
    $.get({
        url: "/detect",
        data: {
            code: code
        },
        success: function(language) {
            $('#detection-result').text(language)
        }
    })
}

$(document).ready(function() {
    // for test
    $("#code").val('def test():\n   print("something")');
});
