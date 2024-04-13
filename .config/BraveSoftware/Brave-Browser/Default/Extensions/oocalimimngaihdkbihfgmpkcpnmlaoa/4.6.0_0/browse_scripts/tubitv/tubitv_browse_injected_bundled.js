/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
var __webpack_exports__ = {};

;// CONCATENATED MODULE: ./src/Teleparty/BrowseScripts/NativePartyHandler.ts
const addNativePartyHandler = (tryAddButton) => {
    setInterval(() => {
        try {
            const buttons = tryAddButton();
            if (buttons) {
                for (const button of buttons) {
                    button.button.addEventListener('click', () => {
                        button.play();
                        const timeout = setTimeout(() => {
                            window.postMessage({
                                type: "startPartyNative",
                            }, "*");
                            clearTimeout(timeout);
                        }, 2000);
                        window.addEventListener("message", (event) => {
                            var _a;
                            if (((_a = event.data) === null || _a === void 0 ? void 0 : _a.type) === "startPartyMessageReceived") {
                                clearTimeout(timeout);
                            }
                        });
                        // Clear the click event listener
                        button.button.removeEventListener('click', () => {
                            //
                        });
                    });
                }
            }
        }
        catch (error) {
            // silent catch
        }
    }, 500);
};

;// CONCATENATED MODULE: ./src/Teleparty/BrowseScripts/TubiTV/tubitv_browse_injected.js

function addNativePartyButton() {
    // It only really makes sense to inject the NPB on TV show pages
    // it would seem. Maybe the front page too, though I'm not sure.
    if (!window.location.href.includes('/series/')) {
        return undefined;
    }
    if (document.getElementById('native-party-button') != null) {
        return undefined;
    }
    const playButton = document.querySelector('button[data-test-id="web-ui-web-button"]');
    if (playButton == null) {
        return undefined;
    }
    const parentDiv = playButton.parentElement;
    const nativePartyButton = document.createElement('button');
    nativePartyButton.setAttribute('class', playButton.getAttribute('class'));
    nativePartyButton.setAttribute('style', 'background: linear-gradient(273.58deg, #9E55A0 0%, #EF3E3A 100%); border-color: #e50914; color: #fff;');
    nativePartyButton.setAttribute('id', 'native-party-button');
    nativePartyButton.innerHTML = '<span>Start a Teleparty</span>';
    parentDiv.insertBefore(nativePartyButton, playButton.nextSibling);
    return [{ button: nativePartyButton, play: () => playButton.click() }];
}
addNativePartyHandler(addNativePartyButton);

/******/ })()
;