function [linelength]=getTwoPointsLength(point0,point1)

linelength = sqrt((point0(1,1)-point1(2,1))^2 + (point0(1,2)-point1(2,2))^2);

end