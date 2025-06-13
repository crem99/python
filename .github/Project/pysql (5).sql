-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Creato il: Set 18, 2024 alle 18:41
-- Versione del server: 10.4.32-MariaDB
-- Versione PHP: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `pysql`
--

-- --------------------------------------------------------

--
-- Struttura della tabella `assenze`
--

CREATE TABLE `assenze` (
  `id` int(255) NOT NULL,
  `medico` int(255) NOT NULL,
  `inizio` date NOT NULL,
  `fine` date NOT NULL,
  `sostituto_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dump dei dati per la tabella `assenze`
--

INSERT INTO `assenze` (`id`, `medico`, `inizio`, `fine`, `sostituto_id`) VALUES
(29, 14, '2024-09-21', '2024-09-22', NULL);

-- --------------------------------------------------------

--
-- Struttura della tabella `calendario`
--

CREATE TABLE `calendario` (
  `data` date NOT NULL,
  `sala_prelievi` int(11) DEFAULT NULL,
  `sala_medicazioni` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dump dei dati per la tabella `calendario`
--

INSERT INTO `calendario` (`data`, `sala_prelievi`, `sala_medicazioni`) VALUES
('2024-09-18', 1, 2),
('2024-09-19', 2, 1),
('2024-09-20', 1, 2),
('2024-09-23', 2, 1),
('2024-09-24', 1, 2),
('2024-09-25', 2, 1),
('2024-09-26', 1, 2),
('2024-09-27', 2, 1);

-- --------------------------------------------------------

--
-- Struttura della tabella `esiti`
--

CREATE TABLE `esiti` (
  `ID` int(11) NOT NULL,
  `medico_id` int(11) DEFAULT NULL,
  `infermiere_id` int(11) DEFAULT NULL,
  `segreteria_id` int(11) DEFAULT NULL,
  `paziente_id` int(11) DEFAULT NULL,
  `esito` text DEFAULT NULL,
  `ambulatorio` varchar(50) DEFAULT NULL,
  `data` date DEFAULT NULL,
  `urgenza` tinyint(1) DEFAULT NULL,
  `tipo_visita` varchar(50) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dump dei dati per la tabella `esiti`
--

INSERT INTO `esiti` (`ID`, `medico_id`, `infermiere_id`, `segreteria_id`, `paziente_id`, `esito`, `ambulatorio`, `data`, `urgenza`, `tipo_visita`) VALUES
(47, 14, NULL, NULL, 65, 'ciao', 'Per Adulti', '2024-09-18', 0, 'Medico curante'),
(49, NULL, 1, NULL, 65, 'kkk', 'Infermieristico', '2024-09-18', 0, 'Infermieristica'),
(50, NULL, 2, NULL, 69, ',yr,irf,u', 'Infermieristico', '2024-09-18', 0, 'Infermieristica'),
(51, NULL, 2, NULL, 65, 'sndrykmd', 'Infermieristico', '2024-09-18', 0, 'Infermieristica'),
(52, NULL, 2, NULL, 69, 'ciao', 'Infermieristico', '2024-09-18', 0, 'Infermieristica'),
(53, 14, NULL, NULL, 69, 'xfmxxkymrd', 'Per Adulti', '2024-09-18', 0, 'Medico curante'),
(54, 14, NULL, NULL, 65, 'dnmrsdnms', 'Per Adulti', '2024-09-18', 0, 'Sostituto');

-- --------------------------------------------------------

--
-- Struttura della tabella `infermieri`
--

CREATE TABLE `infermieri` (
  `id` int(20) NOT NULL,
  `nome` varchar(255) DEFAULT NULL,
  `cognome` varchar(255) DEFAULT NULL,
  `CF` varchar(255) DEFAULT NULL,
  `email` varchar(255) DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dump dei dati per la tabella `infermieri`
--

INSERT INTO `infermieri` (`id`, `nome`, `cognome`, `CF`, `email`, `password`) VALUES
(1, 'Robert', 'Romani', 'crmnflppfie95jdu', 'inf', 'inf'),
(2, 'Mario', 'Rossi', 'crmnflepfie95jdu', 'inf2', 'inf2');

-- --------------------------------------------------------

--
-- Struttura della tabella `medici`
--

CREATE TABLE `medici` (
  `id` int(11) NOT NULL,
  `nome` varchar(255) DEFAULT NULL,
  `cognome` varchar(255) DEFAULT NULL,
  `email` varchar(255) DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL,
  `CF` varchar(255) DEFAULT NULL,
  `specialita` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dump dei dati per la tabella `medici`
--

INSERT INTO `medici` (`id`, `nome`, `cognome`, `email`, `password`, `CF`, `specialita`) VALUES
(14, 'Luigi', 'Verdi', 'luigiverdi@gmail.com', '12345678', 'curmcippodjxge8a', 'Oculista'),
(15, 'Pietr', 'Porpora', 'pietroporpora@gmail.com', '12345678', 'wurmc8336dawplot', 'Oculista');

-- --------------------------------------------------------

--
-- Struttura della tabella `pazienti`
--

CREATE TABLE `pazienti` (
  `id` int(20) NOT NULL,
  `nome` varchar(255) DEFAULT NULL,
  `cognome` varchar(255) DEFAULT NULL,
  `CS` varchar(255) NOT NULL,
  `medico_curante` int(100) DEFAULT NULL,
  `nascita` date DEFAULT NULL,
  `luogo_nascita` varchar(255) DEFAULT NULL,
  `email` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dump dei dati per la tabella `pazienti`
--

INSERT INTO `pazienti` (`id`, `nome`, `cognome`, `CS`, `medico_curante`, `nascita`, `luogo_nascita`, `email`, `password`) VALUES
(65, 'enrico', 'armani', 'crmnclppshrigjal', 14, '1999-12-31', 'cracovia', 'armanienrico@gmail.com', '12345678'),
(67, 'edoardo', 'vantini', 'crmnclssdfgtrfgh', 15, '2021-09-21', 'trent', 'armanienrico@gmail.com', '12345678'),
(69, 'nicola', 'cremon', 'crmncl99c03f861l', 0, '1999-03-03', 'verona', 'nicolacremon@gmail.com', '12345678');

-- --------------------------------------------------------

--
-- Struttura della tabella `prenotazioni`
--

CREATE TABLE `prenotazioni` (
  `ID` int(11) NOT NULL,
  `paziente_id` int(11) DEFAULT NULL,
  `medico_id` int(11) DEFAULT NULL,
  `data` date NOT NULL DEFAULT current_timestamp(),
  `infermiere_id` int(255) DEFAULT NULL,
  `sala` varchar(255) NOT NULL,
  `orario` time NOT NULL,
  `tipo_visita` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dump dei dati per la tabella `prenotazioni`
--

INSERT INTO `prenotazioni` (`ID`, `paziente_id`, `medico_id`, `data`, `infermiere_id`, `sala`, `orario`, `tipo_visita`) VALUES
(54, 69, 14, '2024-09-18', NULL, 'Sala 1', '16:00:00', 'Per adulti'),
(55, 69, NULL, '2024-09-18', 2, 'Sala Medicazioni', '16:15:00', 'Infermieristica'),
(56, 65, 14, '2024-09-18', NULL, 'Sala 1', '17:15:00', 'Per adulti'),
(57, 65, NULL, '2024-09-18', 2, 'Sala Medicazioni', '17:30:00', 'Infermieristica'),
(58, 65, 14, '2024-09-20', NULL, 'Sala 1', '09:00:00', 'Per adulti'),
(59, 65, NULL, '2024-09-20', 2, 'Sala Medicazioni', '09:30:00', 'Infermieristica');

-- --------------------------------------------------------

--
-- Struttura della tabella `responsabilita`
--

CREATE TABLE `responsabilita` (
  `id` int(11) NOT NULL,
  `minorenne_id` int(11) DEFAULT NULL,
  `maggiorenne_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dump dei dati per la tabella `responsabilita`
--

INSERT INTO `responsabilita` (`id`, `minorenne_id`, `maggiorenne_id`) VALUES
(8, 67, 65),
(9, 68, 65);

-- --------------------------------------------------------

--
-- Struttura della tabella `segreteria`
--

CREATE TABLE `segreteria` (
  `id` int(20) NOT NULL,
  `nome` varchar(255) DEFAULT NULL,
  `cognome` varchar(255) DEFAULT NULL,
  `email` varchar(255) DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dump dei dati per la tabella `segreteria`
--

INSERT INTO `segreteria` (`id`, `nome`, `cognome`, `email`, `password`) VALUES
(1, 'patrizia', 'mancini', 'per', 'per');

--
-- Indici per le tabelle scaricate
--

--
-- Indici per le tabelle `assenze`
--
ALTER TABLE `assenze`
  ADD PRIMARY KEY (`id`);

--
-- Indici per le tabelle `calendario`
--
ALTER TABLE `calendario`
  ADD PRIMARY KEY (`data`),
  ADD KEY `fk_sala_prelievi` (`sala_prelievi`),
  ADD KEY `fk_sala_medicazioni` (`sala_medicazioni`);

--
-- Indici per le tabelle `esiti`
--
ALTER TABLE `esiti`
  ADD PRIMARY KEY (`ID`),
  ADD KEY `medico_id` (`medico_id`),
  ADD KEY `infermiere_id` (`infermiere_id`),
  ADD KEY `segreteria_id` (`segreteria_id`),
  ADD KEY `paziente_id` (`paziente_id`);

--
-- Indici per le tabelle `infermieri`
--
ALTER TABLE `infermieri`
  ADD PRIMARY KEY (`id`);

--
-- Indici per le tabelle `medici`
--
ALTER TABLE `medici`
  ADD PRIMARY KEY (`id`);

--
-- Indici per le tabelle `pazienti`
--
ALTER TABLE `pazienti`
  ADD PRIMARY KEY (`id`);

--
-- Indici per le tabelle `prenotazioni`
--
ALTER TABLE `prenotazioni`
  ADD PRIMARY KEY (`ID`),
  ADD KEY `paziente_id` (`paziente_id`),
  ADD KEY `medico_id` (`medico_id`);

--
-- Indici per le tabelle `responsabilita`
--
ALTER TABLE `responsabilita`
  ADD PRIMARY KEY (`id`),
  ADD KEY `minorenne_id` (`minorenne_id`),
  ADD KEY `maggiorenne_id` (`maggiorenne_id`);

--
-- Indici per le tabelle `segreteria`
--
ALTER TABLE `segreteria`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT per le tabelle scaricate
--

--
-- AUTO_INCREMENT per la tabella `assenze`
--
ALTER TABLE `assenze`
  MODIFY `id` int(255) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=30;

--
-- AUTO_INCREMENT per la tabella `esiti`
--
ALTER TABLE `esiti`
  MODIFY `ID` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=55;

--
-- AUTO_INCREMENT per la tabella `infermieri`
--
ALTER TABLE `infermieri`
  MODIFY `id` int(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=18;

--
-- AUTO_INCREMENT per la tabella `medici`
--
ALTER TABLE `medici`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=17;

--
-- AUTO_INCREMENT per la tabella `pazienti`
--
ALTER TABLE `pazienti`
  MODIFY `id` int(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=70;

--
-- AUTO_INCREMENT per la tabella `prenotazioni`
--
ALTER TABLE `prenotazioni`
  MODIFY `ID` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=60;

--
-- AUTO_INCREMENT per la tabella `responsabilita`
--
ALTER TABLE `responsabilita`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;

--
-- AUTO_INCREMENT per la tabella `segreteria`
--
ALTER TABLE `segreteria`
  MODIFY `id` int(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- Limiti per le tabelle scaricate
--

--
-- Limiti per la tabella `calendario`
--
ALTER TABLE `calendario`
  ADD CONSTRAINT `fk_sala_medicazioni` FOREIGN KEY (`sala_medicazioni`) REFERENCES `infermieri` (`id`),
  ADD CONSTRAINT `fk_sala_prelievi` FOREIGN KEY (`sala_prelievi`) REFERENCES `infermieri` (`id`);

--
-- Limiti per la tabella `esiti`
--
ALTER TABLE `esiti`
  ADD CONSTRAINT `esiti_ibfk_1` FOREIGN KEY (`medico_id`) REFERENCES `medici` (`id`),
  ADD CONSTRAINT `esiti_ibfk_2` FOREIGN KEY (`infermiere_id`) REFERENCES `infermieri` (`id`),
  ADD CONSTRAINT `esiti_ibfk_3` FOREIGN KEY (`segreteria_id`) REFERENCES `segreteria` (`id`),
  ADD CONSTRAINT `esiti_ibfk_4` FOREIGN KEY (`paziente_id`) REFERENCES `pazienti` (`id`);

--
-- Limiti per la tabella `prenotazioni`
--
ALTER TABLE `prenotazioni`
  ADD CONSTRAINT `prenotazioni_ibfk_1` FOREIGN KEY (`paziente_id`) REFERENCES `pazienti` (`id`),
  ADD CONSTRAINT `prenotazioni_ibfk_2` FOREIGN KEY (`medico_id`) REFERENCES `medici` (`id`);

--
-- Limiti per la tabella `responsabilita`
--
ALTER TABLE `responsabilita`
  ADD CONSTRAINT `responsabilita_ibfk_2` FOREIGN KEY (`maggiorenne_id`) REFERENCES `pazienti` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
