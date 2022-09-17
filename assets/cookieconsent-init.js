window.addEventListener("load", function () {
// obtain cookieconsent plugin
    var cookieconsent = initCookieConsent();

// run plugin with config object
    cookieconsent.run({
        autorun: true,
        current_lang: "en",
        autoclear_cookies: true,
        page_scripts: true,

        onFirstAction: function (user_preferences, cookie) {
            // callback triggered only once
        },

        onAccept: function (cookie) {
            // ... cookieconsent accepted
        },

        onChange: function (cookie, changed_preferences) {
            // ... cookieconsent preferences were changed/
        },
    })
})
