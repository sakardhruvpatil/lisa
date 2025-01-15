// preload.js

window.addEventListener('DOMContentLoaded', () => {
    // Allow single touch, prevent multi-touch gestures
    document.addEventListener('touchstart', function (e) {
        if (e.touches.length > 1) {
            // Prevent default behavior for multi-touch events
            e.preventDefault();
        }
    }, { passive: false });

    document.addEventListener('touchmove', function (e) {
        if (e.touches.length > 1) {
            // Prevent default behavior during multi-touch movements
            e.preventDefault();
        }
    }, { passive: false });
});