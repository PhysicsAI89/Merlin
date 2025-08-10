DROP DATABASE stockmarket;
CREATE DATABASE stockmarket;
USE stockmarket;

--
-- Table structure for table `stock`
--
CREATE TABLE IF NOT EXISTS `stock` (
  `id`              BIGINT(20)   UNSIGNED NOT NULL AUTO_INCREMENT,
  `symbol`          VARCHAR(10)  DEFAULT NULL,
  `name`            VARCHAR(512) DEFAULT NULL,
  `enabled`         TINYINT(1)   DEFAULT 1  NOT NULL,  
  `advfn_url`       VARCHAR(512) DEFAULT NULL,
  `sector`          VARCHAR(512) DEFAULT '',
  `shares_in_issue` BIGINT(20)   DEFAULT 0,
  PRIMARY KEY (`id`),
  UNIQUE KEY `symbol` (`symbol`)
) ENGINE=INNODB  DEFAULT CHARSET=utf8 AUTO_INCREMENT=1;

--
-- Table structure for table `stock_daily`
--
CREATE TABLE IF NOT EXISTS `stock_daily` (
  `id`          BIGINT(20)   UNSIGNED NOT NULL AUTO_INCREMENT,
  `stock_id`    BIGINT(20)   UNSIGNED NOT NULL,
  `symbol`      VARCHAR(10)  DEFAULT NULL,
  `price_date`  DATE         DEFAULT NULL,
  `open_price`  DOUBLE       DEFAULT 0.00,
  `close_price` DOUBLE       DEFAULT 0.00,
  `high_price`  DOUBLE       DEFAULT 0.00,
  `low_price`   DOUBLE       DEFAULT 0.00,
  `volume`      DOUBLE       DEFAULT 0.00,
  PRIMARY KEY (`id`),
  KEY `stock_id` (`stock_id`),
  KEY `symbol` (`symbol`)
) ENGINE=INNODB  DEFAULT CHARSET=utf8 AUTO_INCREMENT=1;
