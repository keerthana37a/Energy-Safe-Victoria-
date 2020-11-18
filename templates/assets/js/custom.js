/*--------------------------
	baguetteBoxOne map
---------------------------- */	
window.onload = function() {
    if(typeof oldIE === 'undefined' && Object.keys)
        hljs.initHighlighting();

    baguetteBox.run('.baguetteBoxOne');
    baguetteBox.run('.baguetteBoxTwo');
    baguetteBox.run('.baguetteBoxThree', {
        animation: 'fadeIn'
    });
    baguetteBox.run('.baguetteBoxFour', {
        buttons: false
    });
    baguetteBox.run('.baguetteBoxFive', {
        captions: function(element) {
            return element.getElementsByTagName('img')[0].alt;
        }
    });
};



(window.jQuery),

function($) {

    var Page = function() {
        this.$topNavbar = $("#navbar-menu"),
        this.$stickyElem = $("#sticky-nav"),
        this.$backToTop = $('#back-to-top'),
        this.$contactForm = $("#contact-form")
    };
    //
    Page.prototype.init = function () {
        var $this = this;
        //window related event
 
       

        //Handling the scroll event
        $(window).scroll(function(){
            if ($(this).scrollTop() > 100) {
                $this.$backToTop.fadeIn();
            } else {
                $this.$backToTop.fadeOut();
            }
        }); 

        //on click on navbar - Smooth Scroll To Anchor (requires jQuery Easing plugin)
        this.$topNavbar.on('click', function(event) {
            var $anchor = $(event.target);
            if ($($anchor.attr('href')).length > 0 && $anchor.is('a.nav-link')) {
                $('html, body').stop().animate({
                    scrollTop: $($anchor.attr('href')).offset().top - 0
                }, 1500, 'easeInOutExpo');
                event.preventDefault();    
            }
        });

        //back-to-top button
        $this.$backToTop.click(function(){
            $("html, body").animate({ scrollTop: 0 }, 1000);
            return false;
        });


        //init contact app if contact form added in a page
        if(this.$contactForm.length>0)
            $.ContactForm.init();

    },
    //init
    $.Page = new Page, $.Page.Constructor = Page
}(window.jQuery),

//initializing
function($) {
    "use strict";
    $.Page.init()
}(window.jQuery);


/* ---------------------------------------------- /*
	 * Preloader
	/* ---------------------------------------------- */


    "use strict"; // Start of use strict
    // jQuery for page scrolling feature - requires jQuery Easing plugin
    $('a.page-scroll').bind('click', function(event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: ($($anchor.attr('href')).offset().top - 50)
        }, 1250, 'easeInOutExpo');
        event.preventDefault();
    });
    // Highlight the top nav as scrolling occurs
    $('body').scrollspy({
        target: '.navbar-fixed-top',
        offset: 51
    })
    // Offset for Main Navigation
    $('#mainNav').affix({
        offset: {
            top: 100
        }
    })
	


