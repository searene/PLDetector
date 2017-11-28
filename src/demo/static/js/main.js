function detect() {

    var detectionResult = $('#detection-result');
    detectionResult.text('...');

    var textarea = document.getElementById("code");
    var code = textarea.value;
    $.get({
        url: "/detect",
        data: {
            code: code
        },
        success: function(language) {
            detectionResult.text(language)
        }
    })
}

$(document).ready(function() {
    // for test
    $("#code").val('def test():\n   print("something")');
});
