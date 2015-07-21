var jsPlumbSettings = {
    connector: ['Bezier', {curviness:20}],
    paintStyle: {strokeStyle:'red', lineWidth:2},
    endpoint: "Blank",
    anchors: ['Left', 'Right'],
    overlays: [["Label", {cssClass: "label", label: ""}],["PlainArrow", {width: 6, length: 5, id: "arrow", location: -1}]]
};

var jsPlumbSettingsDecisions = {
    connector: ['Bezier', {curviness:20}],
    paintStyle: {strokeStyle:'blue', lineWidth:2},
    endpoint: "Blank",
    anchors: ['Left', 'Right'],
    overlays: [["PlainArrow", {width: 6, length: 5, id: "arrow", location: -1}]]
};

var CortVisualisation = {

    init: function () {
        "use strict";
        this.errors = window.errors;
        this.chain_to_colour = window.chain_to_colour;
        this.jsPlumb = window.jsPlumb;
        var highlightErrors = [];
        this.errors.forEach(function(error) {
            if (error.hasOwnProperty("highlight")) {
                highlightErrors.push(error);
            }
        });
        if (highlightErrors.length > 0) {
            this.setDoc("#" + highlightErrors[0].anaphor.slice(0, highlightErrors[0].anaphor.lastIndexOf("_")));
            var elements = $();
            var docId = this.docId;
            highlightErrors.forEach(function(error){
                var antMentionClass = $("#" + error.antecedent).attr("class").split(" ")[0];
                var anaMentionClass = $("#" + error.anaphor).attr("class").split(" ")[0];
                elements = elements.add($(docId + " #" + error.antecedent + ", " + docId + " #" + error.anaphor + ", " + docId + " .navcontainer ." + antMentionClass + ", " + docId + " .navcontainer ." + anaMentionClass));
            });
            this.highlightElements(elements);
            this.showErrors(highlightErrors);
        } else {
            this.setDoc("#" + $( "#documentsNavi" ).find( "li:first" ).html());
        }
        $("#document_name").text(this.docId.substring(1));
    },

    setDoc: function (docId) {
        "use strict";

        this.docId = docId;
        var docInNavi = $();
        $("#documentsNavi").find("li").each(function() {
            if ($(this).text() === docId.slice(1)) {
                $(this).addClass("highlight");
        		docInNavi = docInNavi.add($(this));
            }
        });
        $(".document").hide();
        $(docId).show();
        this.scrollTo(docInNavi);
    },

    getNeighbouringMentions: function (elements) {
        "use strict";
        var neighbours = $();
        var docId = this.docId;
        elements.each(function(){
            var that = this;
            $(this).parents(".mention").add($(this).children(".mention")).each(function(){               
                if ($(this).attr("data-span") === $(that).attr("data-span")){
                    neighbours = neighbours.add($(this)).add($(docId + " .navcontainer ." + $(this).attr("class").split(" ")[0]));
                }
            });
        });

        return neighbours;
    },

    highlightElements: function (elements) {
        "use strict";

        this.clearHighlight();
        elements = elements || $(this.docId).find(".mention");
        elements = elements.add(this.getNeighbouringMentions(elements));
        var that = this;
        elements.each(function() {
            var mentionClass = $(this).attr("class").split(" ")[0];
            $(this).css("background-color", that.chain_to_colour[mentionClass]);
            if ($(this).attr("class").indexOf("gold") === 0){
                $(this).addClass("goldBorder");
            } else {
                $(this).addClass("blueBorder");
            }
        });
        this.scrollTo(elements.first());
    },

    scrollTo: function (element) {
        "use strict";

        var animateSettings = {
            duration: 1000,
            specialEasing: {
                width: 'linear',
                height: 'easeOutBounce'
            }
        };
        var scroll_to;
        if (element.hasClass("mention")){
            scroll_to = Math.max(0, element.offset().top - $(window).height() + element.height());
            $("body, html").animate(
                {
                    scrollTop: scroll_to
                },
                animateSettings);
        } else if (!element.isOnScreen()){
            scroll_to = Math.max(0, element.offset().top - $(window).height() + element.height());
            $("body, html").animate(
                {
                    scrollTop: scroll_to
                },
                animateSettings);
        }
    },

    clearHighlight: function () {
        "use strict";

        this.jsPlumb.detachEveryConnection();
        $(this.docId).find(".goldNavi li, .systemNavi li, .mention").removeAttr("style").removeClass("goldBorder").removeClass("blueBorder");
    },

    showErrors: function (errors) {
        "use strict";

        var that = this;
        errors.forEach(function(error){
            // Multi-line span hack
            var t_ant_bis = document.createElement("span");
            t_ant_bis.id = error.antecedent + "bis";
            $(t_ant_bis).insertAfter("#" + error.antecedent);
            var t_ana_bis = document.createElement("span");
            t_ana_bis.id = error.anaphor + "bis";
            $(t_ana_bis).insertAfter("#" + error.anaphor);
            if (error.type === "Decision"){
                that.jsPlumb.connect({
                    source: t_ana_bis,
                    target: t_ant_bis
                }, jsPlumbSettingsDecisions);
            }
        });

        errors.forEach(function(error){
            // Multi-line span hack
            var t_ant_bis = document.createElement("span");
            t_ant_bis.id = error.antecedent + "bis";
            $(t_ant_bis).insertAfter("#" + error.antecedent);
            var t_ana_bis = document.createElement("span");
            t_ana_bis.id = error.anaphor + "bis";
            $(t_ana_bis).insertAfter("#" + error.anaphor);
            if (error.type !== "Decision"){
                jsPlumbSettings.overlays[0][1].label = error.type;
                that.jsPlumb.connect({
                    source: t_ana_bis,
                    target: t_ant_bis
                }, jsPlumbSettings);
            }
        });        
    }
};

$(function (){
    "use strict";

    // Mouse events

    // Mentions in text and entities in navigations
    $(".navcontainer .goldNavi li, .navcontainer .systemNavi li, .mention").on({
        click: function (event) {
            event.stopPropagation();
            var mentionClass = $(this).attr("class").split(" ")[0];
            var elements = $(cort.docId).find("." + mentionClass);
            elements = elements.add(cort.getNeighbouringMentions(elements));
            var errors = [];
            cort.errors.forEach(function(error){
                if (elements.index($("#" + error.antecedent)) !== -1 &&
                    elements.index($("#" + error.anaphor)) !== -1){
                    errors.push(error);
                } else if (elements.index($("#" + error.antecedent)) !== -1){
                    elements = elements.add($("#" + error.anaphor));
                    errors.push(error);
                } else if (elements.index($("#" + error.anaphor)) !== -1){
                    elements = elements.add($("#" + error.antecedent));
                    errors.push(error);
                }
            });
            cort.highlightElements(elements);
            if (errors.length > 0) {
                cort.showErrors(errors);
            }
        }
    });

    // Collect all system/gold mentions via the right-side navigation headings
    $(".navcontainer .goldNavi h3, .navcontainer .systemNavi h3").on({
        click: function (event) {
            event.stopPropagation();
            var elements = $();
            $(this).parent().find("li").each(function(){
                elements = elements.add($(cort.docId).find("." + $(this).attr("class").split(" ")[0]));
            });
            cort.highlightElements(elements);
        }
    });

    // Highlight errors
    // By category
    $(".navcontainer .errorsNavi ul li").on({
        click: function(event) {
            event.stopPropagation();
            var errorType = $(this).parent().parent().find("h4").text().split(" ")[0];
            var category = $(this).text().split(" ").slice(0, -1).join(" ").slice(0,-1);
            var elements = $();
            var errors = cort.errors.filter(function(error){
                if (error.type === errorType &&
                    error.antecedent.indexOf(cort.docId.substr(1)) !== -1 &&
                    error.category === category){
                    elements = elements.add($("#" + error.antecedent + ", #" + error.anaphor));
                    return true;
                } else {
                    return false;
                }
            });

            cort.highlightElements(elements);
            cort.showErrors(errors);
        }
    });

    // By error type
    $(".navcontainer .errorsNavi div h4").on({
        click: function(event) {
            event.stopPropagation();
            var errorType = $(this).text().split(" ")[0];
            var errCount = $(this).text().split(" ")[1].substr(1).split(")")[0];
            if (errCount > 0){
                var elements = $();
                var errors = cort.errors.filter(function(error){
                    if (error.type === errorType && error.antecedent.indexOf(cort.docId.substr(1)) !== -1){
                        elements = elements.add($("#" + error.antecedent + ", #" + error.anaphor));
                        return true;
                    } else {
                        return false;
                    }
                });
                cort.highlightElements(elements);
                cort.showErrors(errors);
            }
        }
    });

    // All errors
    $(".navcontainer .errorsNavi h3").on({
        click: function(event) {
            event.stopPropagation();
            var errCount = $(this).text().split(" ")[1].substr(1).split(")")[0];
            if (errCount > 0){
                var elements = $();
                var errors = cort.errors.filter(function(error){
                    if (error.antecedent.indexOf(cort.docId.substr(1)) !== -1){
                        elements = elements.add($("#" + error.antecedent + ", #" + error.anaphor));
                        return true;
                    } else {
                        return false;
                    }
                });
                cort.highlightElements(elements);
                cort.showErrors(errors);
            }
        }
    });


    // Select document
    $("#documentsNavi li").on({
        click: function (event) {
            event.stopPropagation();
            $("#documentsNavi li").removeClass("highlight");
            var doc = "#" + $(this).html();
            cort.setDoc(doc);
            cort.clearHighlight();
            $("#document_name").text($(this).html());
        }
    });

    // jQuery helper methods

    $.fn.isOnScreen = function() {
        var nestingElement, viewport, bounds;
        viewport = {}; // Can either be window or other overflowing/scroll area
        bounds = {};
        if (this.hasClass("mention")){ // mention in text
            nestingElement = $(window);
            viewport.top = $(nestingElement).scrollTop();
            viewport.bottom = viewport.top + $(nestingElement).height();
            bounds.top = this.offset().top;
            bounds.bottom = bounds.top + this.height();
        } else if (this.parents().hasClass("documentsNavi") || // documents navigation element
            this.parents().hasClass("navcontainer")){ // coreference chains navigations
            nestingElement = this.parent();
            viewport.top = 0;
            viewport.bottom = $(nestingElement).height();
            bounds.top = this.offset().top - $(nestingElement).offset().top;
            bounds.bottom = bounds.top + this.outerHeight();
        }
        var visible = (bounds.top <= viewport.bottom) && (bounds.bottom >= viewport.top);

        return visible;

    };

    $(window).resize(function(){
        cort.jsPlumb.repaintEverything();
    });

    var cort = CortVisualisation;
    cort.init();


});