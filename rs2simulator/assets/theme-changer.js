function modifyOffcanvas() {
    console.log("modifyOffcanvas()");

    if (document.readyState !== "loading") {
        setAttributes();
    } else {
        document.addEventListener("DOMContentLoaded", setAttributes);
    }

    console.log("modifyOffcanvas()2");
}

function setAttributes(e) {
    // console.log("event fired:", e);
    //
    // let elements = document.getElementsByClassName("offcanvas-title");
    // console.log(elements, elements.length);
    //
    // console.log("children", elements.children);
    //
    // console.log("first elem", elements[0]);
    //
    // let title = elements.item(0);
    // title.id = "offcanvas-title-label";
    // console.log(title);
}
