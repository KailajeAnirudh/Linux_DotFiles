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

;// CONCATENATED MODULE: ./src/Teleparty/BrowseScripts/Paramount/paramount_browse_injected.js

function setNativePartyButtonStyle(nativePartyButton) {
    const leftMargin = window.innerWidth < 370 ? '0px' : '10px';
    const topMargin = window.innerWidth < 370 ? '10px' : '0px';
    nativePartyButton.setAttribute('style', `background: linear-gradient(273.58deg, #9E55A0 0%, #EF3E3A 100%); border-color: #e50914; color: #fff; margin-left: ${leftMargin}; margin-top: ${topMargin};`);
}
function addNativePartyButton() {
    if (document.getElementById('native-party-button') != null) {
        return undefined;
    }
    const playButton = document.querySelector('.button.playIcon');
    if (playButton == null) {
        return undefined;
    }
    const parentDiv = playButton.parentElement;
    const nativePartyButton = document.createElement('button');
    nativePartyButton.setAttribute('class', playButton.getAttribute('class'));
    setNativePartyButtonStyle(nativePartyButton);
    nativePartyButton.setAttribute('id', 'native-party-button');
    nativePartyButton.innerHTML = '<div class="button__text">Start a Teleparty</div>';
    parentDiv.insertBefore(nativePartyButton, playButton.nextSibling);
    window.addEventListener('resize', () => setNativePartyButtonStyle(nativePartyButton));
    return [{ button: nativePartyButton, play: () => playButton.click() }];
}
addNativePartyHandler(addNativePartyButton);

/******/ })()
;